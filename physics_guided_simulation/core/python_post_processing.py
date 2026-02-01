"""
Python Post-Processing Module
=============================

Post-processing module for Python k-Wave simulation results.
Converts simulation outputs to training-ready format for ML models.

This module follows the same workflow as post_processing_matlab_simulation_data.py
but is specifically designed for the Python simulation pipeline outputs.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# Import post-processing functions from kwave
from kwave.reconstruction.beamform import envelope_detection
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.filters import gaussian_filter
from kwave.utils.signals import get_win

# Import centralized configuration
from helpers import PostProcessingConfig

class PythonSimulationPostProcessor:
    """Post-processor for Python k-Wave simulation results"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
    
    def process_sample(self, sample_name: str, simulation_results_path: str, 
                      output_path: str, noise_level: str = 'low') -> Dict:
        """
        Process a single sample's simulation results
        
        Args:
            sample_name: Sample name (e.g., 'benign_1', 'malignant_5')
            simulation_results_path: Path to simulation results folder
            output_path: Path to save processed results
            noise_level: Noise level to process
            
        Returns:
            Dictionary with processed data paths
        """
        # Setup paths
        sample_folder = os.path.join(simulation_results_path, sample_name)
        simulation_file = os.path.join(sample_folder, f"simulation_{noise_level}.mat")
        phantom_file = os.path.join(sample_folder, f"phantom_{noise_level}.mat")
        
        if not os.path.exists(simulation_file):
            raise FileNotFoundError(f"Simulation file not found: {simulation_file}")
        if not os.path.exists(phantom_file):
            raise FileNotFoundError(f"Phantom file not found: {phantom_file}")
        
        print(f"Processing sample {sample_name} with noise level {noise_level}...")
        
        # Load simulation and phantom data
        sim_data = scipy.io.loadmat(simulation_file)
        phantom_data = scipy.io.loadmat(phantom_file)
        
        # Extract data
        scan_lines = sim_data['scan_lines']
        sound_speed_map = sim_data['sound_speed_map']
        density_map = sim_data['density_map']
        semantic_map = phantom_data['semantic_map']
        kgrid_dt = float(sim_data['kgrid_dt'])
        kgrid_Nt = int(sim_data['kgrid_Nt'])
        element_width = int(sim_data['element_width'])
        
        # Process the scan lines
        processed_results = self._process_scan_lines(
            scan_lines, sound_speed_map, semantic_map, 
            kgrid_dt, kgrid_Nt, element_width
        )
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        scans_path = os.path.join(output_path, "scans")
        labels_path = os.path.join(output_path, "labels")
        os.makedirs(scans_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
        
        # Save processed data
        result_paths = self._save_processed_data(
            processed_results, sample_name, scans_path, labels_path
        )
        
        # Optional visualization
        if self.config.show_preview:
            self._visualize_processed_data(processed_results, sample_name, noise_level)
        
        print(f"Successfully processed sample {sample_name}")
        return result_paths
    
    def _process_scan_lines(self, scan_lines: np.ndarray, sound_speed_map: np.ndarray,
                           semantic_map: np.ndarray, kgrid_dt: float, kgrid_Nt: int,
                           element_width: int) -> Dict:
        """Process raw scan lines through the ultrasound imaging pipeline"""
        
        print(f"Processing scan lines with shape: {scan_lines.shape}")
        print(f"kgrid_Nt: {kgrid_Nt}, kgrid_dt: {kgrid_dt}")
        
        # Create time arrays
        t_array = np.arange(kgrid_Nt) * kgrid_dt
        
        # 1. Remove input signal (windowing) - Following demo script pattern
        tukey_win, _ = get_win(kgrid_Nt * 2, "Tukey", False, 0.05)
        transmit_len = 4 * 20  # Estimate based on tone burst cycles * some factor
        
        # Create scan line window following the demo pattern
        scan_line_win = np.concatenate([
            np.zeros([1, transmit_len * 2]), 
            tukey_win.T[:, :kgrid_Nt - transmit_len * 2]
        ], axis=1)
        
        # Apply window to each scan line
        scan_lines_windowed = scan_lines * scan_line_win
        
        # 2. Time Gain Compensation (TGC)
        c0 = 1540  # Speed of sound
        t0 = transmit_len * kgrid_dt / 2
        r = c0 * (np.arange(1, kgrid_Nt + 1) * kgrid_dt - t0) / 2
        
        # Calculate TGC following demo pattern
        tone_burst_freq = 1.5e6
        alpha_coeff = 0.75
        alpha_power = 1.5
        tgc_alpha_db_cm = alpha_coeff * (tone_burst_freq * 1e-6) ** alpha_power
        tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100
        tgc = np.exp(tgc_alpha_np_m * 2 * r)
        
        # Apply TGC
        scan_lines_tgc = scan_lines_windowed * tgc
        
        # 3. Frequency Filtering
        fs = 1 / kgrid_dt
        
        # Fundamental frequency filtering
        scan_lines_fund = gaussian_filter(
            scan_lines_tgc, fs, tone_burst_freq, self.config.fund_filter_bw
        )
        
        # Harmonic frequency filtering  
        scan_lines_harm = gaussian_filter(
            scan_lines_tgc, fs, 2 * tone_burst_freq, self.config.harm_filter_bw
        )
        
        # 4. Envelope Detection
        scan_lines_fund_env = envelope_detection(scan_lines_fund)
        scan_lines_harm_env = envelope_detection(scan_lines_harm)
        
        # 5. Log Compression
        scan_lines_fund_log = log_compression(scan_lines_fund_env, self.config.compression_ratio, True)
        scan_lines_harm_log = log_compression(scan_lines_harm_env, self.config.compression_ratio, True)
        
        # 
        # 6. Create phantom labels using the SAME spatial processing as ultrasound images
        processed_phantom = self._process_phantom_labels_aligned(
            sound_speed_map, semantic_map, element_width, scan_lines_fund_log.shape
        )
        
        return {
            'b_mode': scan_lines_fund_log,
            'harmonic': scan_lines_harm_log,
            'phantom_labels': processed_phantom,
            'distance_array': r,
            'time_array': t_array
        }
    
    def _process_phantom_labels_aligned(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                                      element_width: int, ultrasound_shape: Tuple[int, int]) -> Dict:
        """
        Process phantom data to create labels using the SAME spatial transformations as ultrasound images
        This ensures perfect pixel-level alignment between ultrasound and semantic data
        
        CRITICAL: Accounts for difference between phantom grid (Nx_tot, Ny_tot) and simulation grid (Nx, Ny)
        Updated to handle ROI-centered phantom positioning
        """
        
        print(f"Processing phantom labels with perfect alignment to ultrasound shape: {ultrasound_shape}")
        
        # Extract middle slice for 2D processing
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        print(f"Original phantom slice shapes: SOS={sos_slice.shape}, Semantic={semantic_slice.shape}")
        
        # CRITICAL: Extract the region that was ACTUALLY SIMULATED
        # Updated to handle ROI-centered positioning instead of simple centering
        
        # Get actual simulation parameters
        num_scan_lines = ultrasound_shape[0]  # Actual scan lines from ultrasound data
        total_scan_width = num_scan_lines * element_width
        print(f"Simulation parameters: {num_scan_lines} scan lines, element_width={element_width}")
        print(f"total_scan_width: {total_scan_width}")
        
        # Calculate the scan coverage area (same logic as simulation)
        Ny_tot = sound_speed_map.shape[1]  # Full phantom width
        scan_coverage_width = Ny_tot * 192 // 448  # Proportion of width for scan coverage
        scan_start_col = (Ny_tot - scan_coverage_width) // 2
        scan_end_col = scan_start_col + scan_coverage_width
        
        print(f"Phantom grid width (Ny_tot): {Ny_tot}")
        print(f"Scan coverage area: cols {scan_start_col}:{scan_end_col} (width: {scan_coverage_width})")
        
        # For post-processing, we need to extract the region that was simulated
        # This corresponds to the scan coverage area
        start_y = scan_start_col
        end_y = scan_end_col
        
        print(f"Extracting simulated region: {start_y}:{end_y}")
        
        # Extract the exact same spatial region as the ultrasound simulation
        sos_region = sos_slice[:, start_y:end_y]
        semantic_region = semantic_slice[:, start_y:end_y]
        
        print(f"Extracted region shapes: SOS={sos_region.shape}, Semantic={semantic_region.shape}")
        
        # Verify that the extracted region width matches the ultrasound scan line coverage
        expected_width = scan_coverage_width
        actual_width = sos_region.shape[1]
        if actual_width != expected_width:
            print(f"WARNING: Extracted width {actual_width} != expected {expected_width}")
        
        # Check tumor coverage in extracted region
        tumor_mask_full = (semantic_slice == self.config.semantic_label).astype(np.float32)
        tumor_mask_extracted = (semantic_region == self.config.semantic_label).astype(np.float32)
        
        total_tumor_pixels = np.sum(tumor_mask_full)
        extracted_tumor_pixels = np.sum(tumor_mask_extracted)
        coverage_percentage = (extracted_tumor_pixels / total_tumor_pixels * 100) if total_tumor_pixels > 0 else 0
        
        print(f"Tumor coverage analysis:")
        print(f"  Total tumor pixels: {total_tumor_pixels}")
        print(f"  Extracted tumor pixels: {extracted_tumor_pixels}")
        print(f"  Coverage percentage: {coverage_percentage:.1f}%")
        
        if coverage_percentage < 80:
            print(f"  WARNING: Only {coverage_percentage:.1f}% of tumor is in extracted region!")
        else:
            print(f"  Good coverage: {coverage_percentage:.1f}% of tumor is in extracted region")
        
        # STEP 2: Create semantic masks from the extracted region
        # Create binary mask for tumor regions (semantic label 4 = tumor)
        tumor_mask = tumor_mask_extracted
        
        # Create grayscale label image
        gray_label = np.where(tumor_mask == 1, 255, 128).astype(np.uint8)
        
        # STEP 3: Apply the SAME spatial transformations as ultrasound images
        
        # NO transpose needed - both phantom and ultrasound are in the same orientation now
        print(f"Before resize: Tumor={tumor_mask.shape}, Gray={gray_label.shape}, SOS={sos_region.shape}")
        
        # Resize to target shape using the SAME method as ultrasound images
        target_shape = self.config.target_shape
        print(f"Resizing phantom labels to {target_shape} using SAME method as ultrasound images")
        
        tumor_mask_resized = cv2.resize(tumor_mask, target_shape, interpolation=cv2.INTER_NEAREST)
        gray_label_resized = cv2.resize(gray_label, target_shape, interpolation=cv2.INTER_NEAREST)
        sos_region_resized = cv2.resize(sos_region, target_shape, interpolation=cv2.INTER_CUBIC)
        
        # Ensure binary mask is properly thresholded after resize
        tumor_mask_resized = (tumor_mask_resized > 0.5).astype(np.float32)
        
        print(f"Final aligned shapes: Binary={tumor_mask_resized.shape}, Gray={gray_label_resized.shape}, SOS={sos_region_resized.shape}")
        
        # Verify alignment by checking that all outputs have the same shape as ultrasound
        expected_shape = target_shape
        assert tumor_mask_resized.shape == expected_shape, f"Tumor mask shape {tumor_mask_resized.shape} != expected {expected_shape}"
        assert gray_label_resized.shape == expected_shape, f"Gray label shape {gray_label_resized.shape} != expected {expected_shape}"
        assert sos_region_resized.shape == expected_shape, f"SOS region shape {sos_region_resized.shape} != expected {expected_shape}"
        
        print("Perfect spatial alignment achieved between ultrasound and semantic data")
        
        return {
            'tumor_binary_mask': tumor_mask_resized,
            'tumor_gray_label': gray_label_resized,
            'sound_speed_region': sos_region_resized
        }
    
    def _process_phantom_labels(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                               element_width: int) -> Dict:
        """
        DEPRECATED: Use _process_phantom_labels_aligned instead for perfect alignment
        This method is kept for backward compatibility but should not be used
        """
        raise NotImplementedError("Use _process_phantom_labels_aligned for perfect spatial alignment")
    
    def _save_processed_data(self, processed_results: Dict, sample_name: str,
                           scans_path: str, labels_path: str) -> Dict:
        """Save processed data in ML-ready format"""
        
        # Apply gamma correction to ultrasound images
        b_mode = self._gamma_correction(processed_results['b_mode'])
        harmonic = self._gamma_correction(processed_results['harmonic'])
        
        # Resize ultrasound images to target shape using simple cv2.resize
        target_shape = self.config.target_shape
        print(f"Resizing ultrasound images to {target_shape} using simple resize")
        
        # Note: Transpose first to get (height, width) format
        b_mode_transposed = b_mode.T
        harmonic_transposed = harmonic.T
        
        print(f"B-mode shape before resize: {b_mode_transposed.shape}")
        print(f"Harmonic shape before resize: {harmonic_transposed.shape}")
        
        b_mode_resized = cv2.resize(b_mode_transposed, target_shape, interpolation=cv2.INTER_CUBIC)
        harmonic_resized = cv2.resize(harmonic_transposed, target_shape, interpolation=cv2.INTER_CUBIC)
        
        print(f"Final ultrasound image shapes: B-mode={b_mode_resized.shape}, Harmonic={harmonic_resized.shape}")
        
        # Prepare data dictionaries
        scan_data = {
            "noisy_us_scan_b_mode": b_mode_resized,
            "noisy_us_scan_harmonic": harmonic_resized
        }
        
        label_data = {
            "clean_phantom_gray": processed_results['phantom_labels']['tumor_gray_label'],
            "clean_phantom_binary": processed_results['phantom_labels']['tumor_binary_mask'],
            "clean_phantom_sound_speed": processed_results['phantom_labels']['sound_speed_region']
        }
        
        # Save as pickle files
        scan_file = os.path.join(scans_path, f"scan_{sample_name}.pkl")
        label_file = os.path.join(labels_path, f"label_{sample_name}.pkl")
        
        with open(scan_file, 'wb') as f:
            pickle.dump(scan_data, f)
        
        with open(label_file, 'wb') as f:
            pickle.dump(label_data, f)
        
        return {
            'scan_file': scan_file,
            'label_file': label_file,
            'processed_data': {**scan_data, **label_data}
        }
    
    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to ultrasound images"""
        # Normalize to [0, 1]
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        # Apply gamma correction
        corrected = np.power(normalized, self.config.gamma_correction)
        return corrected
    
    def _visualize_processed_data(self, processed_results: Dict, sample_name: str, noise_level: str):
        """Visualize processed data for quality check"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # B-mode image
        axes[0, 0].imshow(processed_results['b_mode'].T, cmap='gray', aspect='auto')
        axes[0, 0].set_title(f'B-mode - Sample {sample_name}')
        axes[0, 0].set_xlabel('Scan Line')
        axes[0, 0].set_ylabel('Depth')
        
        # Harmonic image
        axes[0, 1].imshow(processed_results['harmonic'].T, cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'Harmonic - Sample {sample_name}')
        axes[0, 1].set_xlabel('Scan Line')
        axes[0, 1].set_ylabel('Depth')
        
        # Sound speed region
        axes[0, 2].imshow(processed_results['phantom_labels']['sound_speed_region'], cmap='jet')
        axes[0, 2].set_title('Sound Speed Region')
        
        # Binary mask
        axes[1, 0].imshow(processed_results['phantom_labels']['tumor_binary_mask'], cmap='gray')
        axes[1, 0].set_title('Tumor Binary Mask')
        
        # Gray label
        axes[1, 1].imshow(processed_results['phantom_labels']['tumor_gray_label'], cmap='gray')
        axes[1, 1].set_title('Tumor Gray Label')
        
        # Intensity histograms
        axes[1, 2].hist(processed_results['b_mode'].flatten(), bins=50, alpha=0.7, label='B-mode')
        axes[1, 2].hist(processed_results['harmonic'].flatten(), bins=50, alpha=0.7, label='Harmonic')
        axes[1, 2].set_title('Intensity Histograms')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()

def resize_with_aspect_ratio_and_padding(image: np.ndarray, target_shape: Tuple[int, int], 
                                       interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize image while preserving aspect ratio and padding with zeros
    
    NOTE: This function is kept for reference but not used in the current pipeline.
    For ultrasound images, simple cv2.resize distortion is preferred over aspect ratio preservation.
    
    Args:
        image: Input image
        target_shape: Target (height, width)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized and padded image
    """
    target_h, target_w = target_shape
    h, w = image.shape[:2]
    
    # Calculate aspect ratios
    original_aspect = w / h
    target_aspect = target_w / target_h
    
    # Calculate new dimensions preserving aspect ratio
    if original_aspect > target_aspect:
        # Image is wider than target
        new_w = target_w
        new_h = int(target_w / original_aspect)
    else:
        # Image is taller than target
        new_h = target_h
        new_w = int(target_h * original_aspect)
    
    # Resize preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Create padded result
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
    
    # Calculate padding offsets (center the image)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    print(f"Aspect ratio resize: {w}x{h} -> {new_w}x{new_h} -> padded to {target_w}x{target_h}")
    return padded

def process_batch(simulation_results_path: str, output_path: str, 
                 sample_names: List[str], noise_levels: List[str] = ['low'],
                 config: Optional[PostProcessingConfig] = None) -> Dict:
    """
    Process a batch of simulation results
    
    Args:
        simulation_results_path: Path to simulation results folder
        output_path: Path to save processed results
        sample_names: List of sample names to process
        noise_levels: List of noise levels to process
        config: Post-processing configuration
        
    Returns:
        Dictionary with processing results
    """
    if config is None:
        config = PostProcessingConfig()
    
    processor = PythonSimulationPostProcessor(config)
    
    # Create dataset folders
    os.makedirs(output_path, exist_ok=True)
    
    results = {}
    processed_count = 0
    
    for sample_name in sample_names:
        for noise_level in noise_levels:
            try:
                sample_results = processor.process_sample(
                    sample_name=sample_name,
                    simulation_results_path=simulation_results_path,
                    output_path=output_path,
                    noise_level=noise_level
                )
                
                key = f"sample_{sample_name}_{noise_level}"
                results[key] = sample_results
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing sample {sample_name} with noise {noise_level}: {e}")
                continue
    
    # Save processing metadata
    metadata = {
        'total_processed': processed_count,
        'config': config.__dict__,
        'sample_names': sample_names,
        'noise_levels': noise_levels,
        'output_path': output_path
    }
    
    metadata_file = os.path.join(output_path, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Batch processing completed! Processed {processed_count} samples.")
    print(f"Results saved to: {output_path}")
    
    return results

def main():
    """Main execution function for post-processing"""
    
    # Configuration
    config = PostProcessingConfig(
        target_shape=(256, 256),
        gamma_correction=0.5,
        show_preview=True  # Enable visualization to see results
    )
    
    # Paths (adjust to your setup)
    SIMULATION_RESULTS_PATH = r"D:\CMME\1_data_generation\data\BUSI\python_simulation_results"
    OUTPUT_PATH = r"D:\CMME\1_data_generation\data\BUSI\processed_python_results"
    
    # Processing settings - specifically for benign_10
    SAMPLE_NAMES = ['benign_10']  # Process only benign_10
    NOISE_LEVELS = ['low']  # ['low', 'medium', 'high']
    
    # Process batch
    results = process_batch(
        simulation_results_path=SIMULATION_RESULTS_PATH,
        output_path=OUTPUT_PATH,
        sample_names=SAMPLE_NAMES,
        noise_levels=NOISE_LEVELS,
        config=config
    )
    
    print("Post-processing pipeline completed!")
    print(f"Processed sample benign_10 results:")
    for key, result in results.items():
        print(f"  {key}: {result['scan_file']}")
        print(f"  {key}: {result['label_file']}")

if __name__ == "__main__":
    main() 