"""
Python k-Wave Simulation Pipeline
=================================

This script replicates the MATLAB ultrasound simulation workflow using python-kwave.
It follows the same conceptual flow as working_pipeline_in_matlab.m but in Python.

Key features:
- Mask/Bezier-based phantom generation with acoustic perturbations
- Multiple noise levels
- k-Wave ultrasound simulation
- Progress tracking
- Modular and configurable design
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from typing import Dict, Tuple, Optional, List
import cv2

# k-Wave imports
from kwave.data import Vector # type:ignore
from kwave.kgrid import kWaveGrid # type:ignore
from kwave.kmedium import kWaveMedium # type:ignore
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D # type:ignore
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple # type:ignore
from kwave.options.simulation_execution_options import SimulationExecutionOptions # type:ignore
from kwave.options.simulation_options import SimulationOptions # type:ignore
from kwave.utils.signals import tone_burst # type:ignore 
from kwave.utils.dotdictionary import dotdict # type:ignore
from scipy.ndimage import convolve1d

# Import centralized configuration
from helpers import SimulationConfig


class GaussianRandomFieldGenerator:
    """Generate Gaussian Random Fields for tissue modeling"""
    
    @staticmethod
    def create_gaussian_random_field(Nx: int, Ny: int, Nz: int, 
                                   sigma: float, kernel_size: int,
                                   coherence_level: str = 'very_high') -> np.ndarray:
        """
        Create a Gaussian Random Field for tissue heterogeneity modeling
        
        Args:
            Nx, Ny, Nz: Grid dimensions
            sigma: Standard deviation for Gaussian kernel
            kernel_size: Size of the Gaussian kernel
            coherence_level: Level of spatial coherence
            
        Returns:
            3D GRF array
        """
        # Create random noise
        noise = np.random.randn(Nx, Ny, Nz)
        
        # Create Gaussian kernel for filtering
        kernel_1d = GaussianRandomFieldGenerator._gaussian_kernel_1d(kernel_size, sigma)
        
        # Apply coherence based on level
        coherence_map = {
            'low': 0.3,
            'medium': 0.5, 
            'high': 0.7,
            'very_high': 0.9
        }
        coherence_factor = coherence_map.get(coherence_level, 0.9)
        
        # Apply spatial filtering
        
        # Filter along each dimension
        grf = noise
        grf = convolve1d(grf, kernel_1d, axis=0, mode='nearest')
        grf = convolve1d(grf, kernel_1d, axis=1, mode='nearest')
        if Nz > 1:
            grf = convolve1d(grf, kernel_1d, axis=2, mode='nearest')
        
        # Apply coherence factor
        grf = coherence_factor * grf + (1 - coherence_factor) * noise
        
        # Normalize to [0, 1]
        grf = (grf - np.min(grf)) / (np.max(grf) - np.min(grf))
        
        return grf
    
    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
        """Create 1D Gaussian kernel"""
        x = np.arange(size) - size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / np.sum(kernel)

class BezierPhantomGenerator:
    """Generate phantoms based on cubic Bezier curves"""
    
    @staticmethod
    def make_bezier_scattering_region_cubic(
        Nx_tot: int, Ny_tot: int, Nz_tot: int,
        dx: float, dy: float, dz: float,
        P0: List[float], P1: List[float], P2: List[float], P3: List[float],
        physical_radius: float,
        scattering_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a scattering region around a 3D cubic Bezier curve
        
        Args:
            Nx_tot, Ny_tot, Nz_tot: Number of grid points in x, y, and z directions
            dx, dy, dz: Grid spacing in x, y, and z directions (meters)
            P0, P1, P2, P3: Control points of the 3D cubic Bezier curve (in grid points)
            physical_radius: Physical radius (in meters) of the scattering region around the curve
            scattering_map: Initial scattering map to modify (typically zeros)
            
        Returns:
            Tuple of (scattering_map, bezier_curve)
        """
        # Define the parameter t for the Bezier curve (from 0 to 1)
        t = np.linspace(0, 1, 100)  # 100 points along the Bezier curve
        
        # Generate the cubic Bezier curve in 3D using the formula
        # B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        bezier_curve_x = ((1 - t)**3 * P0[0] + 
                         3 * (1 - t)**2 * t * P1[0] + 
                         3 * (1 - t) * t**2 * P2[0] + 
                         t**3 * P3[0])
        bezier_curve_y = ((1 - t)**3 * P0[1] + 
                         3 * (1 - t)**2 * t * P1[1] + 
                         3 * (1 - t) * t**2 * P2[1] + 
                         t**3 * P3[1])
        bezier_curve_z = ((1 - t)**3 * P0[2] + 
                         3 * (1 - t)**2 * t * P1[2] + 
                         3 * (1 - t) * t**2 * P2[2] + 
                         t**3 * P3[2])
        
        # Combine x, y, and z coordinates into the Bezier curve array
        bezier_curve = np.column_stack([bezier_curve_x, bezier_curve_y, bezier_curve_z])
        
        # Convert physical radius to grid points
        radius = round(physical_radius / dx)  # Convert to grid points based on grid spacing
        
        print(f"Generating Bezier phantom with radius {radius} grid points ({physical_radius:.4f} m)")
        print(f"Bezier curve has {len(t)} points")
        
        # Create a copy of the scattering map
        scattering_map = scattering_map.copy()
        
        # Iterate over the grid points and check if the point lies within the radius of the Bezier curve
        # This is computationally intensive, so we'll use vectorized operations where possible
        for i in range(Nx_tot):
            for j in range(Ny_tot):
                for k in range(Nz_tot):
                    # Calculate the distance from each point on the Bezier curve
                    # Using vectorized numpy operations for efficiency
                    distances = np.sqrt(
                        (bezier_curve[:, 0] - i)**2 + 
                        (bezier_curve[:, 1] - j)**2 + 
                        (bezier_curve[:, 2] - k)**2
                    )
                    min_distance = np.min(distances)
                    
                    # If the point is within the defined radius of the Bezier curve, mark it
                    if min_distance <= radius:
                        scattering_map[i, j, k] = 1
        
        print(f"Bezier phantom generation complete. Non-zero points: {np.sum(scattering_map)}")
        return scattering_map, bezier_curve
    
    @staticmethod
    def define_phantom_bezier(
        c0: float, rho0: float,
        Nx_tot: int, Ny_tot: int, Nz_tot: int,
        dx: float, dy: float, dz: float,
        background_map_mean: float, background_map_std: float,
        radius: float,
        x_shift: float, y_shift: float, z_shift: float,
        scattering_mean_c1: float,
        scattering_std_c2: float,
        scattering_divider_c3: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Define a phantom based on a Bezier curve with specified acoustic properties
        
        Args:
            c0: Base speed of sound [m/s]
            rho0: Base density [kg/m^3]
            Nx_tot, Ny_tot, Nz_tot: Grid dimensions
            dx, dy, dz: Grid spacing [m]
            background_map_mean: Mean for background scattering
            background_map_std: Std deviation for background scattering
            radius: Physical radius of the Bezier phantom [m]
            x_shift, y_shift, z_shift: Shifts for control points positioning
            scattering_mean_c1: Constant to add to scattering region sound speed
            scattering_std_c2: Multiplier for scattering region variability
            scattering_divider_c3: Divider for density calculation
            
        Returns:
            Tuple of (sound_speed_map, density_map, scattering_bezier_map)
        """
        # Initialize maps
        noise_free_map = np.zeros((Nx_tot, Ny_tot, Nz_tot))
        
        # Define acoustic perturbations (MATLAB-style Gaussian perturbation model)
        epsilon_bg = np.random.randn(Nx_tot, Ny_tot, Nz_tot)
        epsilon_scatter = np.random.randn(Nx_tot, Ny_tot, Nz_tot)
        
        # Background perturbation: c0_ref * (mean + std * epsilon)
        background_perturbation = background_map_mean + background_map_std * epsilon_bg
        
        # Lesion perturbation: c0_ref + c1 + c2 * epsilon, with density coupled by divider
        scattering_c0 = c0 + scattering_mean_c1 + scattering_std_c2 * epsilon_scatter
        scattering_rho0 = scattering_c0 / scattering_divider_c3
        
        # Initialize uniform medium with background perturbation
        sound_speed_map = c0 * np.ones((Nx_tot, Ny_tot, Nz_tot)) * background_perturbation
        density_map = rho0 * np.ones((Nx_tot, Ny_tot, Nz_tot)) * background_perturbation
        
        # Ensure the radius is within the max allowed limit
        max_radius = 8  # Maximum radius allowed (in meters)
        phantom_radius = min(radius, max_radius)
        
        # Define the control points for the Bezier curve
        # Position of the Bezier curve in the center of the grid
        center_X = Nx_tot / 2
        center_Y = Ny_tot / 2
        center_Z = Nz_tot / 2
        
        # Define control points for cubic Bezier curve
        P0 = [center_X - x_shift, center_Y - y_shift, center_Z - z_shift]
        P1 = [center_X + x_shift/0.5, center_Y - y_shift/0.25, center_Z + z_shift/2]
        P2 = [center_X - x_shift/4, center_Y + y_shift/4, center_Z - z_shift/4]
        P3 = [center_X + x_shift, center_Y + y_shift, center_Z + z_shift]
        
        print(f"Bezier control points:")
        print(f"  P0: {P0}")
        print(f"  P1: {P1}")
        print(f"  P2: {P2}")
        print(f"  P3: {P3}")
        
        # Generate the phantom using Bezier curve
        scattering_bezier_map, bezier_curve = BezierPhantomGenerator.make_bezier_scattering_region_cubic(
            Nx_tot, Ny_tot, Nz_tot, dx, dy, dz,
            P0, P1, P2, P3,
            phantom_radius, noise_free_map
        )
        
        # Modify sound speed and density for the Bezier region
        sound_speed_map[scattering_bezier_map == 1] = scattering_c0[scattering_bezier_map == 1]
        density_map[scattering_bezier_map == 1] = scattering_rho0[scattering_bezier_map == 1]
        
        return sound_speed_map, density_map, scattering_bezier_map

class TissuePropertiesManager:
    """Manage tissue acoustic properties and noise configurations"""
    
    def __init__(self):
        self.tissue_properties = {
            'background': {'sos': 1600, 'density': 1000},
            'fatty': {'sos': 1600, 'density': 1000},
            'glandular': {'sos': 1570, 'density': 1040},
            'tumor': {'sos': 1480, 'density': 1060}
        }
        
        self.noise_configs = {
            'low': {
                'background': 0.0001,
                'fatty': 0.0003,
                'glandular': 0.0002,
                'tumor': 0.00005
            },
            'medium': {
                'background': 0.005,
                'fatty': 0.015,
                'glandular': 0.01,
                'tumor': 0.002
            },
            'high': {
                'background': 0.02,
                'fatty': 0.06,
                'glandular': 0.04,
                'tumor': 0.01
            }
        }
    
    def create_tissue_maps(self, tumor_mask_3d: np.ndarray, grf: np.ndarray, 
                          noise_level: str = 'low') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sound speed and density maps with tissue-specific noise
        
        Args:
            tumor_mask_3d: 3D binary mask for tumor regions
            grf: Gaussian random field for tissue classification
            noise_level: Level of noise to apply
            
        Returns:
            Tuple of (sound_speed_map, density_map, semantic_map)
        """
        # Use the CORRECT dimensions from the input arrays (Nx_tot, Ny_tot, Nz_tot)
        Nx_tot, Ny_tot, Nz_tot = tumor_mask_3d.shape
        print(f"Creating tissue maps for grid: {Nx_tot} x {Ny_tot} x {Nz_tot}")
        
        # Create tissue type masks based on GRF thresholding (like MATLAB)
        threshold = 0.5
        fatty_mask = (grf < threshold) & ~tumor_mask_3d
        glandular_mask = (grf >= threshold) & ~tumor_mask_3d
        
        # Background mask should be for regions where there's no tissue modeling
        # In MATLAB, background_mask = zeros, but ALL non-tumor regions get tissue properties
        # So we'll model ALL non-tumor regions as either fatty or glandular
        background_mask = np.zeros_like(tumor_mask_3d, dtype=bool)  # Keep minimal background
        
        print(f"Tissue mask statistics:")
        print(f"  Tumor pixels: {np.sum(tumor_mask_3d)} ({100*np.sum(tumor_mask_3d)/tumor_mask_3d.size:.2f}%)")
        print(f"  Fatty pixels: {np.sum(fatty_mask)} ({100*np.sum(fatty_mask)/tumor_mask_3d.size:.2f}%)")
        print(f"  Glandular pixels: {np.sum(glandular_mask)} ({100*np.sum(glandular_mask)/tumor_mask_3d.size:.2f}%)")
        print(f"  Background pixels: {np.sum(background_mask)} ({100*np.sum(background_mask)/tumor_mask_3d.size:.2f}%)")
        
        # Create semantic map for visualization
        semantic_map = np.zeros((Nx_tot, Ny_tot, Nz_tot))
        semantic_map[background_mask] = 1  # Background (should be minimal)
        semantic_map[fatty_mask] = 2       # Fatty tissue
        semantic_map[glandular_mask] = 3   # Glandular tissue
        semantic_map[tumor_mask_3d] = 4    # Tumor/cyst
        
        # Initialize property maps with background values (ensure float64 for noise multiplication)
        sound_speed_map = np.full((Nx_tot, Ny_tot, Nz_tot), self.tissue_properties['background']['sos'], dtype=np.float64)
        density_map = np.full((Nx_tot, Ny_tot, Nz_tot), self.tissue_properties['background']['density'], dtype=np.float64)
        
        print(f"Initialized maps with background properties:")
        print(f"  Sound speed: {self.tissue_properties['background']['sos']} m/s")
        print(f"  Density: {self.tissue_properties['background']['density']} kg/m³")
        
        # Apply tissue properties in order (like MATLAB)
        # Fatty tissue
        if np.any(fatty_mask):
            props = self.tissue_properties['fatty']
            sound_speed_map[fatty_mask] = float(props['sos'])
            density_map[fatty_mask] = float(props['density'])
            print(f"Applied fatty tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # Glandular tissue
        if np.any(glandular_mask):
            props = self.tissue_properties['glandular']
            sound_speed_map[glandular_mask] = float(props['sos'])
            density_map[glandular_mask] = float(props['density'])
            print(f"Applied glandular tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # Tumor tissue (applied last to override)
        if np.any(tumor_mask_3d):
            props = self.tissue_properties['tumor']
            sound_speed_map[tumor_mask_3d] = float(props['sos'])
            density_map[tumor_mask_3d] = float(props['density'])
            print(f"Applied tumor tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # Apply noise to each tissue region (like MATLAB)
        noise_std = self.noise_configs[noise_level]
        print(f"Applying noise (level: {noise_level}):")
        
        # Apply noise to each tissue type
        for tissue_type, mask in [
            ('fatty', fatty_mask),
            ('glandular', glandular_mask), 
            ('tumor', tumor_mask_3d)
        ]:
            if np.any(mask):
                std = noise_std[tissue_type]
                noise_multiplier = np.random.normal(1.0, std, (Nx_tot, Ny_tot, Nz_tot))
                sound_speed_map[mask] *= noise_multiplier[mask]
                density_map[mask] *= noise_multiplier[mask]
                print(f"  {tissue_type}: std={std}, pixels={np.sum(mask)}")
        
        # Apply background noise to remaining regions (very minimal)
        background_regions = ~(fatty_mask | glandular_mask | tumor_mask_3d)
        if np.any(background_regions):
            std = noise_std['background']
            noise_multiplier = np.random.normal(1.0, std, (Nx_tot, Ny_tot, Nz_tot))
            sound_speed_map[background_regions] *= noise_multiplier[background_regions]
            density_map[background_regions] *= noise_multiplier[background_regions]
            print(f"  background: std={std}, pixels={np.sum(background_regions)}")
        
        # Print final statistics
        print(f"Final property map statistics:")
        print(f"  Sound speed range: {np.min(sound_speed_map):.1f} - {np.max(sound_speed_map):.1f} m/s")
        print(f"  Density range: {np.min(density_map):.1f} - {np.max(density_map):.1f} kg/m³")
        
        # Save debug visualization of speed of sound map
        self._save_debug_sos_visualization(sound_speed_map, semantic_map, noise_level)
        
        return sound_speed_map, density_map, semantic_map
    
    def _save_debug_sos_visualization(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray, noise_level: str):
        """Save simple debug visualization of speed of sound map"""
        
        # Extract middle slice
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        # Create simple visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speed of sound map
        im1 = axes[0].imshow(sos_slice, cmap='jet', aspect='auto')
        axes[0].set_title(f'Speed of Sound Map - {noise_level}\nRange: {np.min(sos_slice):.0f} - {np.max(sos_slice):.0f} m/s')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Depth (pixels)')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Speed [m/s]')
        
        # Semantic map
        im2 = axes[1].imshow(semantic_slice, cmap='viridis', vmin=1, vmax=4, aspect='auto')
        axes[1].set_title(f'Tissue Types - {noise_level}')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Depth (pixels)')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_ticks([1, 2, 3, 4])
        cbar2.set_ticklabels(['Background', 'Fatty', 'Glandular', 'Tumor'])
        
        plt.tight_layout()
        
        # Save to debug folder
        debug_folder = "debug_visualizations"
        os.makedirs(debug_folder, exist_ok=True)
        save_path = os.path.join(debug_folder, f'debug_sos_map_{noise_level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DEBUG: Speed of sound map saved to {save_path}")

class PhantomResizer:
    """Utility class for resizing phantoms while preserving aspect ratio and ensuring ROI coverage"""
    
    @staticmethod
    def resize_preserve_aspect_ratio(input_image: np.ndarray, target_height: int, 
                                   target_width: int, resize_method: str = 'nearest') -> np.ndarray:
        """
        Resize an image while preserving aspect ratio (MATLAB-compatible implementation)
        
        This function resizes an input image to fit within target dimensions while
        preserving the original aspect ratio. If necessary, it pads the result to
        exactly match the target dimensions.
        
        Args:
            input_image: Original image to resize
            target_height: Target height (rows) - Nx_tot
            target_width: Target width (columns) - Ny_tot  
            resize_method: Interpolation method ('nearest', 'bilinear', 'bicubic')
            
        Returns:
            Resized and padded image matching target dimensions
        """
        # Get original dimensions
        rows, cols = input_image.shape[:2]
        original_aspect_ratio = cols / rows
        target_aspect_ratio = target_width / target_height
        
        print(f"Original phantom dimensions: {rows} x {cols} (aspect ratio: {original_aspect_ratio:.3f})")
        print(f"Target grid dimensions: {target_height} x {target_width} (aspect ratio: {target_aspect_ratio:.3f})")
        
        # Calculate new dimensions that preserve aspect ratio
        if original_aspect_ratio > target_aspect_ratio:
            # Original is wider than target
            new_width = target_width
            new_height = round(new_width / original_aspect_ratio)
            if new_height > target_height:
                new_height = target_height
                new_width = round(new_height * original_aspect_ratio)
        else:
            # Original is taller than target
            new_height = target_height
            new_width = round(new_height * original_aspect_ratio)
            if new_width > target_width:
                new_width = target_width
                new_height = round(new_width / original_aspect_ratio)
        
        print(f"Resized dimensions (aspect preserved): {new_height} x {new_width}")
        
        # Resize preserving aspect ratio
        if resize_method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif resize_method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif resize_method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_NEAREST
        
        resized_aspect_preserved = cv2.resize(input_image, (new_width, new_height), interpolation=interpolation)
        
        # Create padded version to match target dimensions (centered)
        #         # % Create padded version to match target dimensions
        # resized_image = zeros(target_height, target_width);
        # start_row = round((target_height - new_height) / 2) + 1;
        # start_col = round((target_width - new_width) / 2) + 1;
        # resized_image(start_row:start_row+new_height-1, start_col:start_col+new_width-1) = resized_aspect_preserved;

        resized_image = np.zeros((target_height, target_width), dtype=input_image.dtype)
        start_row = round((target_height - new_height) / 2) 
        start_col = round((target_width - new_width) / 2)
        
        print(f"Phantom positioned at rows {start_row}:{start_row + new_height}, cols {start_col}:{start_col + new_width}")
        
        resized_image[start_row:start_row + new_height, start_col:start_col + new_width] = resized_aspect_preserved
        
        return resized_image
    
    @staticmethod
    def position_phantom_roi_aware(input_image: np.ndarray, target_height: int, target_width: int, 
                                 scan_coverage_info: dict, tumor_threshold: int = 100, 
                                 resize_method: str = 'nearest') -> np.ndarray:
        """
        Position phantom ensuring tumor ROI is well within scan coverage area
        
        Args:
            input_image: Original phantom image
            target_height: Target grid height (Nx_tot)
            target_width: Target grid width (Ny_tot)
            scan_coverage_info: Dictionary with scan coverage parameters
            tumor_threshold: Pixel value threshold for tumor detection
            resize_method: Interpolation method
            
        Returns:
            Positioned phantom with tumor guaranteed within scan coverage
        """
        print(f"\nSIMPLE PHANTOM POSITIONING")
        print(f"=" * 40)
        print(f"Original phantom: {input_image.shape[0]} x {input_image.shape[1]}")
        print(f"Target grid: {target_height} x {target_width}")
        print(f"Scan coverage width: {scan_coverage_info['scan_coverage_width']}")
        
        # Step 1: Resize phantom to fit exactly within scan coverage width
        # cv2.resize takes (width, height) = (cols, rows)
        mask_2d_resized = cv2.resize(input_image, (scan_coverage_info['scan_coverage_width'], target_height), interpolation=cv2.INTER_NEAREST)
        print(f"Resized phantom to: {mask_2d_resized.shape[0]} x {mask_2d_resized.shape[1]} (rows x cols)")
        
        # Step 2: Create full canvas and center the phantom in Y direction (columns)
        canvas = np.zeros((target_height, target_width), dtype=input_image.dtype)
        
        # Center in Y direction (columns) - place phantom in middle of grid width
        center_y = target_width // 2  # Center column position
        y1 = center_y - scan_coverage_info['scan_coverage_width'] // 2  # Left edge
        y2 = y1 + scan_coverage_info['scan_coverage_width']  # Right edge
        
        print(f"Canvas: {canvas.shape[0]} x {canvas.shape[1]} (rows x cols)")
        print(f"Placing phantom at columns {y1}:{y2} (center at {center_y})")
        
        # Step 3: Place resized phantom in center of canvas
        canvas[:, y1:y2] = mask_2d_resized
        mask_2d_resized = canvas
        
        # Verify phantom positioning
        tumor_mask_check = (mask_2d_resized == tumor_threshold)
        if np.any(tumor_mask_check):
            tumor_rows, tumor_cols = np.where(tumor_mask_check)
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            tumor_center_col = (tumor_min_col + tumor_max_col) / 2
            
            print(f"- POSITIONING VERIFICATION:")
            print(f"  Final phantom shape: {mask_2d_resized.shape[0]} x {mask_2d_resized.shape[1]}")
            print(f"  Tumor spans columns: {tumor_min_col}:{tumor_max_col}")
            print(f"  Tumor center column: {tumor_center_col:.1f}")
            print(f"  Scan coverage: {scan_coverage_info['scan_start_col']}:{scan_coverage_info['scan_end_col']}")
            print(f"  Tumor within scan: {'- YES' if tumor_min_col >= scan_coverage_info['scan_start_col'] and tumor_max_col <= scan_coverage_info['scan_end_col'] else '- NO'}")
            
            # Calculate margins to scan edges
            left_margin = tumor_min_col - scan_coverage_info['scan_start_col']
            right_margin = scan_coverage_info['scan_end_col'] - tumor_max_col
            print(f"  Margins to scan edges: left={left_margin:.1f}, right={right_margin:.1f}")
        else:
            print(f"WARNING: No tumor found after positioning")
        
        return mask_2d_resized

class UltrasoundSimulator:
    """Main ultrasound simulation class using python-kwave"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.tissue_manager = TissuePropertiesManager()
        self.grf_generator = GaussianRandomFieldGenerator()
        
        # Calculate grid parameters using Vector objects
        self.pml_size_points = Vector([config.pml_x_size, config.pml_y_size, config.pml_z_size])
        self.grid_size_points = Vector([config.Nx, config.Ny, config.Nz])
        
        # Calculate grid spacing
        grid_spacing_x = config.x_size / config.Nx
        self.grid_spacing_meters = Vector([grid_spacing_x, grid_spacing_x, grid_spacing_x])
        
        # Setup k-Wave grid using correct API
        self.kgrid = kWaveGrid(self.grid_size_points, self.grid_spacing_meters)
        t_end = (config.Nx * grid_spacing_x) * 2.2 / config.c0
        self.kgrid.makeTime(config.c0, t_end=t_end)
        
        # Setup input signal
        self.input_signal = tone_burst(
            1 / self.kgrid.dt, 
            config.tone_burst_freq, 
            config.tone_burst_cycles
        )
        self.input_signal = (config.source_strength / (config.c0 * config.rho0)) * self.input_signal
        
        # Setup medium
        self.medium = kWaveMedium(
            sound_speed=None,  # will be set for each scan line
            alpha_coeff=config.alpha_coeff,
            alpha_power=config.alpha_power,
            BonA=config.BonA
        )
    
    def setup_transducer(self) -> NotATransducer:
        """Setup the ultrasound transducer"""
        config = self.config
        
        # Create simple transducer - smaller for reduced domain
        transducer_params = dotdict()
        transducer_params.number_elements = 32  # Reduced from 32
        transducer_params.element_width = config.element_width
        # Ensure element length fits within the grid (avoid z overflow in small runs)
        max_element_length = max(1, int(self.grid_size_points.z) - 2)
        transducer_params.element_length = min(24, max_element_length)
        transducer_params.element_spacing = 0
        transducer_params.radius = float("inf")
        
        # Position transducer
        transducer_width = (transducer_params.number_elements * transducer_params.element_width + 
                          (transducer_params.number_elements - 1) * transducer_params.element_spacing)
        
        y_pos = self.grid_size_points.y / 2 - transducer_width / 2
        z_pos = self.grid_size_points.z / 2 - transducer_params.element_length / 2
        y_pos = max(0, min(self.grid_size_points.y - transducer_width, y_pos))
        z_pos = max(0, min(self.grid_size_points.z - transducer_params.element_length, z_pos))

        transducer_params.position = np.round([1, y_pos, z_pos])
        
        simple_transducer = kWaveTransducerSimple(self.kgrid, **transducer_params)
        
        # Create NotATransducer for beamforming
        not_transducer_params = dotdict()
        not_transducer_params.sound_speed = config.c0
        not_transducer_params.focus_distance = 20e-3  # Reduced from 20e-3
        not_transducer_params.elevation_focus_distance = 19e-3  # Reduced from 19e-3
        not_transducer_params.steering_angle = 0
        not_transducer_params.transmit_apodization = "Hanning"
        not_transducer_params.receive_apodization = "Rectangular"
        not_transducer_params.active_elements = np.ones((simple_transducer.number_elements, 1))
        not_transducer_params.input_signal = self.input_signal
        
        return NotATransducer(simple_transducer, self.kgrid, **not_transducer_params)
    
    def load_phantom_mask(self, mask_path: str) -> np.ndarray:
        """Load and prepare phantom mask for simulation with ROI-aware positioning ensuring tumor is well within scan coverage"""
        config = self.config
        
        # Load 2D mask
        mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_2d is None:
            raise FileNotFoundError(f"Could not load mask from {mask_path}")
        
        # Calculate total grid size like MATLAB code:
        # Nx_tot = Nx (simulation grid size in depth direction)
        # Ny_tot = Ny + number_scan_lines * element_width (includes scan line coverage)
        # Nz_tot = Nz (elevation direction)
        Nx_tot = config.Nx  # Depth direction
        Ny_tot = config.Ny + config.number_scan_lines * config.element_width  # Width + scan coverage
        Nz_tot = config.Nz  # Elevation direction
        
        print(f"Calculated grid dimensions:")
        print(f"  Nx_tot (depth): {Nx_tot}")
        print(f"  Ny_tot (width + scan coverage): {Ny_tot}")
        print(f"  Nz_tot (elevation): {Nz_tot}")
        print(f"  Scan coverage: {config.number_scan_lines} scan lines × {config.element_width} width = {config.number_scan_lines * config.element_width}")
        
        # Calculate scan coverage information
        scan_coverage_width = config.number_scan_lines * config.element_width
        scan_start_col = config.Ny // 2
        scan_end_col = Ny_tot - config.Ny // 2
        
        scan_coverage_info = {
            'scan_start_col': scan_start_col,
            'scan_end_col': scan_end_col,
            'scan_coverage_width': scan_coverage_width,
            'total_grid_width': Ny_tot
        }
        
        print(f"Scan coverage info:")
        print(f"  Total grid width: {Ny_tot}")
        print(f"  Scan coverage: cols {scan_start_col}:{scan_end_col} (width: {scan_end_col - scan_start_col})")
        print(f"  Scan lines coverage: {scan_coverage_width}")
        
        # Use simple positioning to ensure tumor is well within scan coverage
        resizer = PhantomResizer()
        mask_2d_resized = resizer.position_phantom_roi_aware(
            input_image=mask_2d,
            target_height=Nx_tot,
            target_width=Ny_tot,
            scan_coverage_info=scan_coverage_info,
            tumor_threshold=100,  # Assumed tumor pixel value
            resize_method='nearest'
        )
        
        # Create 3D mask (tumor regions have value 100 or 255, depending on source)
        # Be more flexible with tumor detection values
        tumor_threshold = 100 # Assumed that the tumor is 100 in the mask
        tumor_mask_3d = np.repeat((mask_2d_resized == tumor_threshold)[:, :, np.newaxis], Nz_tot, axis=2)
        
        print(f"Tumor mask created:")
        print(f"  Unique values in 2D mask: {np.unique(mask_2d_resized)}")
        print(f"  Tumor threshold: {tumor_threshold}")
        print(f"  Tumor pixels found: {np.sum(tumor_mask_3d)}")
        print(f"  Tumor percentage: {100 * np.sum(tumor_mask_3d) / tumor_mask_3d.size:.2f}%")
        
        # Visualize the resized phantom to verify ROI placement
        self._save_phantom_resize_visualization(mask_2d, mask_2d_resized, config, scan_coverage_info)
        
        return tumor_mask_3d, (Nx_tot, Ny_tot, Nz_tot)

    def _create_acoustic_maps_from_mask(
        self,
        tumor_mask_3d: np.ndarray,
        sigma_bg_range: Tuple[float, float] = (0.006, 0.010),
        sigma_lesion_range: Tuple[float, float] = (0.001, 0.100),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Create acoustic maps using the latex perturbation model (no GRF).
        Background: c_ref*(1 + sigma_bg*epsilon_bg), rho_ref*(1 + sigma_bg*epsilon_bg)
        Lesion: c_ref*(1 + sigma_lesion*epsilon_l), rho = c/1.5
        """
        config = self.config
        Nx_tot, Ny_tot, Nz_tot = tumor_mask_3d.shape

        epsilon_bg = np.random.randn(Nx_tot, Ny_tot, Nz_tot)
        epsilon_lesion = np.random.randn(Nx_tot, Ny_tot, Nz_tot)

        sigma_bg = float(np.random.uniform(sigma_bg_range[0], sigma_bg_range[1]))
        sigma_lesion = float(np.random.uniform(sigma_lesion_range[0], sigma_lesion_range[1]))

        background_scatter = sigma_bg * epsilon_bg
        sound_speed_map = config.c0 * (1.0 + background_scatter)
        density_map = config.rho0 * (1.0 + background_scatter)

        lesion_scatter = sigma_lesion * epsilon_lesion
        lesion_c0 = config.c0 * (1.0 + lesion_scatter)

        sound_speed_map[tumor_mask_3d] = lesion_c0[tumor_mask_3d]
        density_map[tumor_mask_3d] = lesion_c0[tumor_mask_3d] / 1.5

        # Minimal semantic map: 1=background, 4=lesion (keep visualization compatibility)
        semantic_map = np.ones_like(tumor_mask_3d, dtype=np.uint8)
        semantic_map[tumor_mask_3d] = 4

        print(f"Acoustic perturbations:")
        print(f"  sigma_bg={sigma_bg:.4f}, sigma_lesion={sigma_lesion:.4f}")
        print(f"  Sound speed range: {np.min(sound_speed_map):.1f} - {np.max(sound_speed_map):.1f} m/s")
        print(f"  Density range: {np.min(density_map):.1f} - {np.max(density_map):.1f} kg/m³")

        return sound_speed_map, density_map, semantic_map, sigma_bg, sigma_lesion
    
    def create_bezier_phantom(self, bezier_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """
        Create a phantom based on Bezier curves instead of loading from BUSI dataset
        
        Args:
            bezier_params: Dictionary with Bezier phantom parameters. If None, uses defaults.
                Expected keys:
                - radius: Physical radius of the phantom [m] (default: 0.005)
                - x_shift, y_shift, z_shift: Control point positioning shifts (default: 20, 20, 10)
                - background_map_mean, background_map_std: Background scattering (default: 1.0, 0.01)
                - scattering_mean_c1: Constant to add to scattering region (default: 50)
                - scattering_std_c2: Multiplier for variability (default: 10)
                - scattering_divider_c3: Divider for density (default: 1.5)
        
        Returns:
            Tuple of (sound_speed_map, density_map, scattering_bezier_map, grid_dims)
        """
        config = self.config
        
        # Calculate total grid size like MATLAB code
        Nx_tot = config.Nx
        Ny_tot = config.Ny + config.number_scan_lines * config.element_width
        Nz_tot = config.Nz
        
        # Get grid spacing
        dx = config.x_size / config.Nx
        dy = dx
        dz = dx
        
        print(f"Creating Bezier phantom with grid dimensions:")
        print(f"  Nx_tot (depth): {Nx_tot}")
        print(f"  Ny_tot (width): {Ny_tot}")
        print(f"  Nz_tot (elevation): {Nz_tot}")
        print(f"  Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        
        # Default parameters if not provided
        if bezier_params is None:
            bezier_params = {}
        
        # Extract parameters with defaults
        radius = bezier_params.get('radius', 0.005)  # 5 mm default
        x_shift = bezier_params.get('x_shift', 20)
        y_shift = bezier_params.get('y_shift', 20)
        z_shift = bezier_params.get('z_shift', 10)
        background_map_mean = bezier_params.get('background_map_mean', 1.0)
        background_map_std = bezier_params.get('background_map_std', 0.01)
        scattering_mean_c1 = bezier_params.get('scattering_mean_c1', 50)
        scattering_std_c2 = bezier_params.get('scattering_std_c2', 10)
        scattering_divider_c3 = bezier_params.get('scattering_divider_c3', 1.5)
        
        print(f"Bezier phantom parameters:")
        print(f"  Radius: {radius} m")
        print(f"  Shifts: x={x_shift}, y={y_shift}, z={z_shift}")
        print(f"  Background: mean={background_map_mean}, std={background_map_std}")
        print(f"  Scattering: c1={scattering_mean_c1}, c2={scattering_std_c2}, c3={scattering_divider_c3}")
        
        # Generate Bezier phantom using the BezierPhantomGenerator
        sound_speed_map, density_map, scattering_bezier_map = BezierPhantomGenerator.define_phantom_bezier(
            c0=config.c0,
            rho0=config.rho0,
            Nx_tot=Nx_tot,
            Ny_tot=Ny_tot,
            Nz_tot=Nz_tot,
            dx=dx,
            dy=dy,
            dz=dz,
            background_map_mean=background_map_mean,
            background_map_std=background_map_std,
            radius=radius,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
            scattering_mean_c1=scattering_mean_c1,
            scattering_std_c2=scattering_std_c2,
            scattering_divider_c3=scattering_divider_c3
        )
        
        print(f"Bezier phantom generated successfully")
        print(f"  Phantom region pixels: {np.sum(scattering_bezier_map)}")
        print(f"  Sound speed range: {np.min(sound_speed_map):.1f} - {np.max(sound_speed_map):.1f} m/s")
        print(f"  Density range: {np.min(density_map):.1f} - {np.max(density_map):.1f} kg/m³")
        
        # Visualize the Bezier phantom
        self._visualize_bezier_phantom(scattering_bezier_map, sound_speed_map, density_map, Nz_tot)
        
        return sound_speed_map, density_map, scattering_bezier_map, (Nx_tot, Ny_tot, Nz_tot)
    
    def _visualize_bezier_phantom(self, scattering_map: np.ndarray, sound_speed_map: np.ndarray, 
                                   density_map: np.ndarray, Nz_tot: int):
        """Visualize the Bezier phantom"""
        z_middle = Nz_tot // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scattering region mask
        axes[0].imshow(scattering_map[:, :, z_middle], cmap='gray', aspect='auto')
        axes[0].set_title('Bezier Phantom Mask\n(middle z-slice)')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Depth (pixels)')
        
        # Sound speed map
        im1 = axes[1].imshow(sound_speed_map[:, :, z_middle], cmap='jet', aspect='auto')
        axes[1].set_title('Sound Speed Map\n(middle z-slice)')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Depth (pixels)')
        cbar1 = plt.colorbar(im1, ax=axes[1])
        cbar1.set_label('Speed [m/s]')
        
        # Density map
        im2 = axes[2].imshow(density_map[:, :, z_middle], cmap='viridis', aspect='auto')
        axes[2].set_title('Density Map\n(middle z-slice)')
        axes[2].set_xlabel('Width (pixels)')
        axes[2].set_ylabel('Depth (pixels)')
        cbar2 = plt.colorbar(im2, ax=axes[2])
        cbar2.set_label('Density [kg/m³]')
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = 'bezier_phantom_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Bezier phantom visualization saved to: {save_path}")
        plt.close()
    
    def _save_phantom_resize_visualization(self, original_mask: np.ndarray, 
                                         resized_mask: np.ndarray, config: SimulationConfig, scan_coverage_info: dict):
        """Save visualization of phantom resizing to verify ROI placement with ROI-aware positioning"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ROI-Aware Phantom Positioning & Scan Coverage Analysis', fontsize=16)
        
        # Original mask
        axes[0, 0].imshow(original_mask, cmap='gray')
        axes[0, 0].set_title(f'Original Mask\n{original_mask.shape[0]}×{original_mask.shape[1]}')
        axes[0, 0].axis('off')
        
        # Add tumor detection to original
        tumor_mask_orig = (original_mask == 100)
        if np.any(tumor_mask_orig):
            tumor_rows, tumor_cols = np.where(tumor_mask_orig)
            tumor_min_row, tumor_max_row = tumor_rows.min(), tumor_rows.max()
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            from matplotlib.patches import Rectangle
            rect_orig = Rectangle((tumor_min_col, tumor_min_row), 
                                tumor_max_col - tumor_min_col, tumor_max_row - tumor_min_row,
                                linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
            axes[0, 0].add_patch(rect_orig)
            axes[0, 0].text(tumor_min_col, tumor_min_row-5, 'Original Tumor', 
                           color='red', fontweight='bold', fontsize=10)
        
        # Resized mask with scan coverage overlay
        axes[0, 1].imshow(resized_mask, cmap='gray', alpha=0.8)
        axes[0, 1].set_title(f'ROI-Aware Positioned Mask\n{resized_mask.shape[0]}×{resized_mask.shape[1]}')
        axes[0, 1].axis('off')
        
        # Get scan coverage info
        scan_start_col = scan_coverage_info['scan_start_col']
        scan_end_col = scan_coverage_info['scan_end_col']
        
        # Draw scan line coverage rectangle
        rect = Rectangle((scan_start_col, 0), scan_end_col - scan_start_col, resized_mask.shape[0], 
                        linewidth=3, edgecolor='blue', facecolor='none', linestyle='-', alpha=0.8)
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(scan_start_col + 5, 15, 'Scan Coverage', 
                       color='blue', fontweight='bold', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add margins visualization
        margin_percentage = 0.05  # 5% margin on each side for simple positioning
        scan_coverage_width = scan_end_col - scan_start_col
        effective_scan_start = scan_start_col + scan_coverage_width * margin_percentage
        effective_scan_end = scan_end_col - scan_coverage_width * margin_percentage
        
        rect_effective = Rectangle((effective_scan_start, 0), 
                                 effective_scan_end - effective_scan_start, resized_mask.shape[0], 
                                 linewidth=2, edgecolor='green', facecolor='none', linestyle='--', alpha=0.8)
        axes[0, 1].add_patch(rect_effective)
        axes[0, 1].text(effective_scan_start + 5, 35, 'Effective Scan Area\n(with margins)', 
                       color='green', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Highlight final tumor position
        tumor_mask_final = (resized_mask == 100)
        if np.any(tumor_mask_final):
            tumor_rows, tumor_cols = np.where(tumor_mask_final)
            tumor_min_row, tumor_max_row = tumor_rows.min(), tumor_rows.max()
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            tumor_center_col = (tumor_min_col + tumor_max_col) / 2
            
            rect_tumor = Rectangle((tumor_min_col, tumor_min_row), 
                                 tumor_max_col - tumor_min_col, tumor_max_row - tumor_min_row,
                                 linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
            axes[0, 1].add_patch(rect_tumor)
            axes[0, 1].text(tumor_min_col, tumor_min_row-10, 'Positioned Tumor', 
                           color='red', fontweight='bold', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # ROI analysis - focused view
        ax = axes[1, 0]
        if np.any(tumor_mask_final):
            # Create focused view around tumor
            tumor_center_row = (tumor_min_row + tumor_max_row) // 2
            tumor_center_col = int(tumor_center_col)
            
            # Define region around tumor
            view_size = 100  # pixels around tumor center
            view_start_row = max(0, tumor_center_row - view_size)
            view_end_row = min(resized_mask.shape[0], tumor_center_row + view_size)
            view_start_col = max(0, tumor_center_col - view_size)
            view_end_col = min(resized_mask.shape[1], tumor_center_col + view_size)
            
            focused_view = resized_mask[view_start_row:view_end_row, view_start_col:view_end_col]
            ax.imshow(focused_view, cmap='Reds')
            ax.set_title(f'Focused Tumor View\nCenter: ({tumor_center_row}, {tumor_center_col:.0f})')
            
            # Add scan coverage lines in focused view
            if scan_start_col >= view_start_col and scan_start_col <= view_end_col:
                ax.axvline(x=scan_start_col - view_start_col, color='blue', linewidth=2, linestyle='-', alpha=0.8, label='Scan Start')
            if scan_end_col >= view_start_col and scan_end_col <= view_end_col:
                ax.axvline(x=scan_end_col - view_start_col, color='blue', linewidth=2, linestyle='-', alpha=0.8, label='Scan End')
            if effective_scan_start >= view_start_col and effective_scan_start <= view_end_col:
                ax.axvline(x=effective_scan_start - view_start_col, color='green', linewidth=2, linestyle='--', alpha=0.8, label='Effective Start')
            if effective_scan_end >= view_start_col and effective_scan_end <= view_end_col:
                ax.axvline(x=effective_scan_end - view_start_col, color='green', linewidth=2, linestyle='--', alpha=0.8, label='Effective End')
            
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No tumor found\nin resized image', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Focused Tumor View')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Depth (pixels)')
        
        # Detailed analysis
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate comprehensive statistics
        analysis_text = []
        analysis_text.append("ROI-AWARE POSITIONING ANALYSIS")
        analysis_text.append("=" * 35)
        
        # Original tumor stats
        if np.any(tumor_mask_orig):
            orig_tumor_pixels = np.sum(tumor_mask_orig)
            analysis_text.append(f"Original tumor pixels: {orig_tumor_pixels}")
            analysis_text.append(f"Original tumor area: {orig_tumor_pixels / (original_mask.shape[0] * original_mask.shape[1]) * 100:.1f}%")
        
        # Final tumor stats
        if np.any(tumor_mask_final):
            final_tumor_pixels = np.sum(tumor_mask_final)
            analysis_text.append(f"Final tumor pixels: {final_tumor_pixels}")
            analysis_text.append(f"Final tumor area: {final_tumor_pixels / (resized_mask.shape[0] * resized_mask.shape[1]) * 100:.1f}%")
            
            # Positioning analysis
            analysis_text.append("")
            analysis_text.append("📍 POSITIONING RESULTS:")
            analysis_text.append(f"Tumor cols: {tumor_min_col}:{tumor_max_col}")
            analysis_text.append(f"Tumor center: {tumor_center_col:.1f}")
            analysis_text.append(f"Scan coverage: {scan_start_col}:{scan_end_col}")
            analysis_text.append(f"Effective scan: {effective_scan_start:.1f}:{effective_scan_end:.1f}")
            
            # Check if tumor is within effective scan area
            tumor_in_effective = (tumor_min_col >= effective_scan_start and tumor_max_col <= effective_scan_end)
            tumor_in_scan = (tumor_min_col >= scan_start_col and tumor_max_col <= scan_end_col)
            
            analysis_text.append("")
            analysis_text.append("- COVERAGE VERIFICATION:")
            analysis_text.append(f"Tumor in scan area: {'- YES' if tumor_in_scan else '- NO'}")
            analysis_text.append(f"Tumor in effective area: {'- YES' if tumor_in_effective else '- NO'}")
            
            # Calculate margins
            left_margin = tumor_min_col - scan_start_col
            right_margin = scan_end_col - tumor_max_col
            effective_left_margin = tumor_min_col - effective_scan_start
            effective_right_margin = effective_scan_end - tumor_max_col
            
            analysis_text.append("")
            analysis_text.append("📏 MARGINS:")
            analysis_text.append(f"To scan edges: L={left_margin:.1f}, R={right_margin:.1f}")
            analysis_text.append(f"To effective edges: L={effective_left_margin:.1f}, R={effective_right_margin:.1f}")
            
            # Success assessment
            analysis_text.append("")
            if tumor_in_effective and effective_left_margin > 10 and effective_right_margin > 10:
                analysis_text.append("🎉 SUCCESS: Tumor well positioned!")
                analysis_text.append("   Tumor within effective scan area")
                analysis_text.append("   Adequate margins on both sides")
            elif tumor_in_scan:
                analysis_text.append("PARTIAL: Tumor in scan area but")
                analysis_text.append("   close to edges")
            else:
                analysis_text.append("ISSUE: Tumor extends outside scan area")
        else:
            analysis_text.append("ERROR: No tumor found in final image")
        
        # Display analysis
        analysis_str = '\n'.join(analysis_text)
        ax.text(0.05, 0.95, analysis_str, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = 'phantom_roi_aware_positioning_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROI-aware positioning analysis saved to: {save_path}")
        plt.close()
        
        # Print summary to console
        if np.any(tumor_mask_final):
            print(f"\nROI-AWARE POSITIONING SUMMARY:")
            print(f"  Tumor position: cols {tumor_min_col}:{tumor_max_col} (center: {tumor_center_col:.1f})")
            print(f"  Scan coverage: cols {scan_start_col}:{scan_end_col}")
            print(f"  Effective scan: cols {effective_scan_start:.1f}:{effective_scan_end:.1f}")
            print(f"  Tumor in effective area: {'- YES' if tumor_in_effective else '- NO'}")
            print(f"  Margins: left={effective_left_margin:.1f}, right={effective_right_margin:.1f}")
            
            if tumor_in_effective and effective_left_margin > 10 and effective_right_margin > 10:
                print(f"  🎉 SUCCESS: Tumor optimally positioned with adequate margins!")
            elif tumor_in_scan:
                print(f"  WARNING: Tumor in scan area but close to edges")
            else:
                print(f"  ERROR: Tumor positioning failed - extends outside scan area")
        else:
            print(f"  ERROR: No tumor found after positioning")
    
    def run_simulation(self, sample_name: str, input_data_path: str, output_path: str,
                      noise_levels: List[str] = ['low'], run_simulation: bool = True) -> Dict:
        """
        Run the complete simulation pipeline for a sample
        
        Args:
            sample_name: Sample name (e.g., 'benign_1', 'malignant_5')
            input_data_path: Path to processed phantom data folder  
            output_path: Path to save simulation results
            noise_levels: List of noise levels to simulate
            run_simulation: Whether to run simulation or just load results
            
        Returns:
            Dictionary with simulation results
        """
        config = self.config
        
        # Setup paths
        sample_folder = os.path.join(input_data_path, sample_name)
        mask_path = os.path.join(sample_folder, "mask.png")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for sample {sample_name}: {mask_path}")
        
        # Create output directory
        results_folder = os.path.join(output_path, sample_name)
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"Processing sample {sample_name}...")
        
        # Load and prepare phantom
        tumor_mask_3d, grid_dims = self.load_phantom_mask(mask_path)
        Nx_tot, Ny_tot, Nz_tot = grid_dims
        
        results = {}
        
        # Process each noise level
        for noise_level in noise_levels:
            print(f"Processing noise level: {noise_level}")
            
            # Create acoustic maps using latex-style perturbation model (no GRF)
            sound_speed_map, density_map, semantic_map, sigma_bg, sigma_lesion = (
                self._create_acoustic_maps_from_mask(tumor_mask_3d)
            )
            
            # Save phantom data
            phantom_file = os.path.join(results_folder, f"phantom_{noise_level}.mat")
            scipy.io.savemat(phantom_file, {
                'sound_speed_map': sound_speed_map,
                'density_map': density_map,
                'semantic_map': semantic_map,
                'tumor_mask_3d': tumor_mask_3d,
                'sigma_bg': sigma_bg,
                'sigma_lesion': sigma_lesion,
                'sample_name': sample_name  # Include sample name for reference
            })
            
            # Visualize phantom
            self._visualize_phantom(sound_speed_map, semantic_map, sample_name, noise_level, results_folder)
            
            if run_simulation:
                # Run k-Wave simulation
                print("Running k-Wave simulation...")
                scan_lines = self._run_kwave_simulation(sound_speed_map, density_map)
                
                # Save simulation results
                simulation_file = os.path.join(results_folder, f"simulation_{noise_level}.mat")
                scipy.io.savemat(simulation_file, {
                    'scan_lines': scan_lines,
                    'sound_speed_map': sound_speed_map,
                    'density_map': density_map,
                    'kgrid_dt': self.kgrid.dt,
                    'kgrid_Nt': self.kgrid.Nt,
                    'element_width': config.element_width,
                    'sample_name': sample_name  # Include sample name for reference
                })
                
                results[noise_level] = {
                    'scan_lines': scan_lines,
                    'phantom_file': phantom_file,
                    'simulation_file': simulation_file
                }
            else:
                results[noise_level] = {
                    'phantom_file': phantom_file
                }
        
        print(f"Completed processing sample {sample_name}")
        return results
    
    def run_simulation_bezier(self, phantom_name: str, output_path: str,
                             bezier_params: Optional[Dict] = None, 
                             run_simulation: bool = True) -> Dict:
        """
        Run simulation pipeline using Bezier curve phantom generation
        
        Args:
            phantom_name: Name for this phantom (e.g., 'bezier_phantom_1')
            output_path: Path to save simulation results
            bezier_params: Parameters for Bezier phantom generation (see create_bezier_phantom)
            run_simulation: Whether to run simulation or just generate phantom
            
        Returns:
            Dictionary with simulation results
        """
        config = self.config
        
        # Create output directory
        results_folder = os.path.join(output_path, phantom_name)
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"Processing Bezier phantom: {phantom_name}...")
        
        # Generate Bezier phantom
        sound_speed_map, density_map, scattering_bezier_map, grid_dims = self.create_bezier_phantom(bezier_params)
        Nx_tot, Ny_tot, Nz_tot = grid_dims
        
        results = {}
        
        # For Bezier phantoms, we don't use GRF tissue classification
        # The phantom is already fully defined with sound speed and density maps
        # We just save and optionally simulate it
        
        # Save phantom data
        phantom_file = os.path.join(results_folder, f"phantom_bezier.mat")
        scipy.io.savemat(phantom_file, {
            'sound_speed_map': sound_speed_map,
            'density_map': density_map,
            'scattering_bezier_map': scattering_bezier_map,
            'phantom_name': phantom_name
        })
        
        print(f"Bezier phantom saved to: {phantom_file}")
        
        if run_simulation:
            # Run k-Wave simulation
            print("Running k-Wave simulation...")
            scan_lines = self._run_kwave_simulation(sound_speed_map, density_map)
            
            # Save simulation results
            simulation_file = os.path.join(results_folder, f"simulation_bezier.mat")
            scipy.io.savemat(simulation_file, {
                'scan_lines': scan_lines,
                'sound_speed_map': sound_speed_map,
                'density_map': density_map,
                'kgrid_dt': self.kgrid.dt,
                'kgrid_Nt': self.kgrid.Nt,
                'element_width': config.element_width,
                'phantom_name': phantom_name
            })
            
            results['bezier'] = {
                'scan_lines': scan_lines,
                'phantom_file': phantom_file,
                'simulation_file': simulation_file
            }
        else:
            results['bezier'] = {
                'phantom_file': phantom_file
            }
        
        print(f"Completed processing Bezier phantom {phantom_name}")
        return results
    
    def _run_kwave_simulation(self, sound_speed_map: np.ndarray, 
                             density_map: np.ndarray) -> np.ndarray:
        """Run the k-Wave simulation for all scan lines"""
        config = self.config
        
        # Setup transducer
        transducer = self.setup_transducer()
        
        # Initialize scan lines storage
        scan_lines = np.zeros((config.number_scan_lines, self.kgrid.Nt))
        
        # Run simulation for each scan line
        medium_position = 0
        
        for scan_line_index in range(config.number_scan_lines):
            print(f"Computing scan line {scan_line_index + 1}/{config.number_scan_lines}")
            
            # Extract medium slice
            end_pos = medium_position + self.grid_size_points.y
            self.medium.sound_speed = sound_speed_map[:, medium_position:end_pos, :]
            self.medium.density = density_map[:, medium_position:end_pos, :]
            
            # Set GPU usage based on config
            use_gpu = config.data_cast == 'gpuArray-single'
            
            # Setup simulation options for this scan line - use 'single' for data_cast always
            input_filename = f"scan_line_{scan_line_index}.h5"
            scan_simulation_options = SimulationOptions(
                pml_inside=False,
                pml_size=self.pml_size_points,
                data_cast='single',  # Always use 'single' - GPU is controlled via execution_options
                data_recast=True,
                save_to_disk=True,  # Required for CPU simulations
                input_filename=input_filename,
                save_to_disk_exit=False
            )
            
            # Run k-Wave simulation
            try:
                sensor_data = kspaceFirstOrder3D(
                    medium=self.medium,
                    kgrid=self.kgrid,
                    source=transducer,
                    sensor=transducer,
                    simulation_options=scan_simulation_options,
                    execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu)
                )
                
                # Extract scan line
                scan_lines[scan_line_index, :] = transducer.scan_line(
                    transducer.combine_sensor_data(sensor_data["p"].T)
                )
                
                # Clean up temporary file
                if os.path.exists(input_filename):
                    os.remove(input_filename)
                
            except Exception as e:
                print(f"Error in scan line {scan_line_index}: {e}")
                # Fill with zeros if simulation fails
                scan_lines[scan_line_index, :] = np.zeros(self.kgrid.Nt)
                
                # Clean up temporary file on error
                if os.path.exists(input_filename):
                    os.remove(input_filename)
            
            # Update position
            medium_position += config.element_width
        
        return scan_lines
    
    def _visualize_phantom(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                          sample_name: str, noise_level: str, output_folder: str):
        """Visualize phantom properties"""
        
        # Extract middle slice
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sound speed map
        im1 = axes[0].imshow(sos_slice, aspect='auto', cmap='jet')
        axes[0].set_title(f'Sound Speed - Sample {sample_name} - {noise_level}')
        axes[0].set_xlabel('Width [pixels]')
        axes[0].set_ylabel('Depth [pixels]')
        plt.colorbar(im1, ax=axes[0], label='Speed [m/s]')
        
        # Semantic map
        im2 = axes[1].imshow(semantic_slice, aspect='auto', cmap='viridis')
        axes[1].set_title(f'Tissue Types - Sample {sample_name}')
        axes[1].set_xlabel('Width [pixels]')
        axes[1].set_ylabel('Depth [pixels]')
        cbar = plt.colorbar(im2, ax=axes[1])
        cbar.set_ticks([1, 2, 3, 4])
        cbar.set_ticklabels(['Background', 'Fatty', 'Glandular', 'Tumor'])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_folder, f'phantom_visualization_{noise_level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    
    # Configuration
    config = SimulationConfig()
    
    # Paths (adjust these to your setup)
    INPUT_DATA_PATH = r"D:\CMME\1_data_generation\data\BUSI\processed_data_v2"
    OUTPUT_PATH = r"D:\CMME\1_data_generation\data\BUSI\python_simulation_results"
    
    # Simulation settings
    SAMPLE_NAMES = ['benign_1', 'malignant_5']  # List of sample names to process
    NOISE_LEVELS = ['low']  # ['low', 'medium', 'high']
    RUN_SIMULATION = True   # Set to False to skip simulation
    
    # Create simulator
    simulator = UltrasoundSimulator(config)
    
    # Process each sample
    all_results = {}
    
    for sample_name in SAMPLE_NAMES:
        try:
            results = simulator.run_simulation(
                sample_name=sample_name,
                input_data_path=INPUT_DATA_PATH,
                output_path=OUTPUT_PATH,
                noise_levels=NOISE_LEVELS,
                run_simulation=RUN_SIMULATION
            )
            all_results[sample_name] = results
            
        except Exception as e:
            print(f"Error processing sample {sample_name}: {e}")
            continue
    
    print("Simulation pipeline completed!")
    print(f"Processed {len(all_results)} samples successfully")

if __name__ == "__main__":
    main() 