"""
Complete Python k-Wave Ultrasound Simulation Pipeline
====================================================

This script orchestrates the complete pipeline:
1. Data preparation (preserving original IDs)
2. k-Wave simulation
3. Post-processing for ML training

Features:
- Integrated WandB logging for experiment tracking
- Comprehensive visualization and metadata logging
- Complete pipeline configuration tracking

Usage:
    python run_python_pipeline.py --config config.json --wandb-project "my-project"

Or modify the configuration directly in this script.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import our custom modules
from core.prepare_real_world_phantoms_v2 import PhantomDataProcessor
from core.python_kwave_simulation_pipeline import UltrasoundSimulator
from core.python_post_processing import process_batch
from helpers.file_utils import (
    get_available_samples,
    get_samples_by_id_range,
    get_samples_by_sequential_range,
    load_json_safely,
    save_json_safely,
)
from helpers import (
    SimulationConfig,
    PostProcessingConfig,
    create_config_from_args,
    BaseOptions,
)

# WandB integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


class PipelineConfig(BaseOptions):
    """Configuration class for the complete pipeline inheriting from BaseOptions"""

    def __init__(self):
        # Initialize base options with all defaults
        super().__init__()

        # Pipeline-specific attributes
        self.sample_names_to_process = None  # None means process all available
        self.wandb_dir = None  # Directory for wandb files (None uses default)
        self.phantom_type = "busi"  # "busi" or "bezier"
        self.bezier_name = "bezier_minimal"
        self.small_simulation = True

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "PipelineConfig":
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class WandBLogger:
    """WandB logging integration for pipeline tracking"""

    def __init__(self, config: PipelineConfig, sim_config: SimulationConfig):
        self.config = config
        self.sim_config = sim_config
        self.wandb_run = None
        self.sample_metadata = {}

    def initialize_wandb(self, run_name: str = None):
        """Initialize WandB run with comprehensive configuration"""
        if not WANDB_AVAILABLE or not self.config.enable_wandb:
            return

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"pipeline_run_{timestamp}_{self.config.max_samples}samples"

        # Set up wandb directory if specified
        wandb_init_kwargs = {
            "project": self.config.wandb_project,
            "entity": self.config.wandb_entity,
            "name": run_name,
            "config": self._create_comprehensive_config(),
            "tags": ["ultrasound", "simulation", "k-wave", "python-pipeline"],
        }

        # Add directory if specified
        if self.config.wandb_dir:
            # Create directory if it doesn't exist
            os.makedirs(self.config.wandb_dir, exist_ok=True)
            wandb_init_kwargs["dir"] = self.config.wandb_dir
            print(f"WandB directory set to: {self.config.wandb_dir}")

        # Initialize WandB
        self.wandb_run = wandb.init(**wandb_init_kwargs)

        print(f"WandB initialized: {self.wandb_run.get_url()}")
        if self.config.wandb_dir:
            print(f"  WandB files will be saved to: {self.config.wandb_dir}")

    def _create_comprehensive_config(self) -> Dict:
        """Create comprehensive configuration for WandB"""
        return {
            # Pipeline Configuration
            "pipeline/max_samples": self.config.max_samples,
            "pipeline/noise_levels": self.config.noise_levels,
            "pipeline/target_phantom_size": self.config.target_phantom_size,
            "pipeline/use_gpu": self.config.use_gpu,
            "pipeline/stages": {
                "data_preparation": self.config.run_data_preparation,
                "simulation": self.config.run_simulation,
                "post_processing": self.config.run_post_processing,
            },
            # Simulation Configuration
            "simulation/domain_size": [
                self.sim_config.Nx,
                self.sim_config.Ny,
                self.sim_config.Nz,
            ],
            "simulation/physical_size_mm": self.sim_config.x_size
            * 1000,  # Convert to mm
            "simulation/pml_size": [
                self.sim_config.pml_x_size,
                self.sim_config.pml_y_size,
                self.sim_config.pml_z_size,
            ],
            "simulation/sound_speed_ms": self.sim_config.c0,
            "simulation/density_kgm3": self.sim_config.rho0,
            "simulation/alpha_coeff": self.sim_config.alpha_coeff,
            "simulation/alpha_power": self.sim_config.alpha_power,
            "simulation/BonA": self.sim_config.BonA,
            # Source/Transducer Configuration
            "transducer/source_strength": self.sim_config.source_strength,
            "transducer/frequency_MHz": self.sim_config.tone_burst_freq / 1e6,
            "transducer/tone_burst_cycles": self.sim_config.tone_burst_cycles,
            "transducer/scan_lines": self.sim_config.number_scan_lines,
            "transducer/element_width": self.sim_config.element_width,
            # GRF Configuration
            "grf/sigma": self.sim_config.grf_sigma,
            "grf/kernel_size": self.sim_config.grf_kernel_size,
            "grf/coherence_level": self.sim_config.coherence_level,
            # Technical Configuration
            "technical/data_cast": self.sim_config.data_cast,
            "technical/scaling_factor": self.sim_config.sc,
            # System Information
            "system/timestamp": datetime.now().isoformat(),
            "system/python_version": sys.version,
        }

    def load_sample_metadata(self, processed_phantoms_path: str):
        """Load sample metadata for mapping original to processed samples"""
        metadata_file = os.path.join(processed_phantoms_path, "sample_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.sample_metadata = json.load(f)
            print(f"Loaded metadata for {len(self.sample_metadata)} samples from: {metadata_file}")
            if self.sample_metadata:
                # Show example of first sample's metadata structure
                first_sample = list(self.sample_metadata.keys())[0]
                first_sample_keys = list(self.sample_metadata[first_sample].keys())
                print(f"  Sample metadata structure (example: {first_sample}): {first_sample_keys}")
        else:
            print(f"WARNING: Metadata file not found: {metadata_file}")
            self.sample_metadata = {}

    def log_sample_experiment(
        self,
        sample_name: str,
        sample_index: int,
        processed_phantoms_path: str,
        simulation_results_path: str,
        final_output_path: str,
        noise_level: str = "low",
    ):
        """Log comprehensive experiment data for a single sample"""

        if not self.wandb_run:
            return

        print(f"Logging sample {sample_name} to WandB...")

        try:
            # Load all data for this sample
            data = self._load_sample_data(
                sample_name,
                processed_phantoms_path,
                simulation_results_path,
                final_output_path,
                noise_level,
            )

            # Create comprehensive visualization
            fig = self._create_sample_visualization(sample_name, data)

            # Log sample-specific configuration
            sample_config = self._create_sample_config(sample_name, data, sample_index)

            # Add individual component images
            component_images = self._create_individual_component_images(
                sample_name, data
            )

            # Extract figures to close before logging
            component_figures = component_images.pop("_figures_to_close", [])

            # Prepare all log data BEFORE logging to avoid file handle issues
            log_data = {
                f"samples/{sample_name}/complete_pipeline": wandb.Image(
                    fig, caption=f"Complete pipeline visualization for {sample_name}"
                ),
                f"samples/{sample_name}/sample_index": sample_index,
                "current_sample": sample_name,
                "processing_progress": (sample_index + 1) / self.config.max_samples,
            }

            # Add component images to log data
            log_data.update(component_images)

            # Add sample configuration
            log_data.update(sample_config)

            # Log everything at once - this ensures WandB processes all images before we close figures
            wandb.log(log_data)

            # Small delay to ensure WandB has processed the images on Windows
            time.sleep(0.1)

            # Now close all figures
            plt.close(fig)

            # Close component figures if they exist
            for component_fig in component_figures:
                plt.close(component_fig)

            print(f"Sample {sample_name} logged to WandB")

        except Exception as e:
            print(f"Warning: Failed to log sample {sample_name} to WandB: {e}")

    def _load_sample_data(
        self,
        sample_name: str,
        processed_phantoms_path: str,
        simulation_results_path: str,
        final_output_path: str,
        noise_level: str,
    ) -> Dict:
        """Load all data for a sample"""
        data = {}

        # Load original BUSI data
        sample_metadata = self.sample_metadata.get(sample_name, {})
        print(f"  Sample metadata keys: {list(sample_metadata.keys()) if sample_metadata else 'No metadata found'}")
        if sample_metadata:
            # Fix: Use correct metadata keys from prepare_real_world_phantoms_v2.py
            original_scan_path = sample_metadata.get("original_image_path", "")
            original_mask_path = sample_metadata.get("original_mask_path", "")

            if original_scan_path and os.path.exists(original_scan_path):
                data["original_scan"] = cv2.imread(
                    original_scan_path, cv2.IMREAD_GRAYSCALE
                )
                print(f"  Loaded original scan from: {original_scan_path}")
            else:
                print(f"  WARNING: Original scan not found: {original_scan_path}")
                
            if original_mask_path and os.path.exists(original_mask_path):
                data["original_mask"] = cv2.imread(
                    original_mask_path, cv2.IMREAD_GRAYSCALE
                )
                print(f"  Loaded original mask from: {original_mask_path}")
            else:
                print(f"  WARNING: Original mask not found: {original_mask_path}")

        # Load processed phantom data
        processed_scan_path = os.path.join(
            processed_phantoms_path, f"{sample_name}.png"
        )
        processed_mask_path = os.path.join(
            processed_phantoms_path, f"{sample_name}_mask.png"
        )

        if os.path.exists(processed_scan_path):
            data["processed_scan"] = cv2.imread(
                processed_scan_path, cv2.IMREAD_GRAYSCALE
            )
        if os.path.exists(processed_mask_path):
            data["processed_mask"] = cv2.imread(
                processed_mask_path, cv2.IMREAD_GRAYSCALE
            )

        # Load simulation results
        simulation_folder = os.path.join(simulation_results_path, sample_name)
        simulation_file = os.path.join(
            simulation_folder, f"simulation_{noise_level}.mat"
        )
        phantom_file = os.path.join(simulation_folder, f"phantom_{noise_level}.mat")

        if os.path.exists(simulation_file):
            sim_data = scipy.io.loadmat(simulation_file)
            data["scan_lines"] = sim_data.get("scan_lines")
            # Fix numpy deprecation warnings by properly extracting scalars
            kgrid_dt_val = sim_data.get("kgrid_dt", 0)
            kgrid_Nt_val = sim_data.get("kgrid_Nt", 0)
            data["kgrid_dt"] = float(
                kgrid_dt_val.item() if hasattr(kgrid_dt_val, "item") else kgrid_dt_val
            )
            data["kgrid_Nt"] = int(
                kgrid_Nt_val.item() if hasattr(kgrid_Nt_val, "item") else kgrid_Nt_val
            )

        if os.path.exists(phantom_file):
            phantom_data = scipy.io.loadmat(phantom_file)
            data["sound_speed_map"] = phantom_data.get("sound_speed_map")
            data["density_map"] = phantom_data.get("density_map")
            data["semantic_map"] = phantom_data.get("semantic_map")

        # Load final processed data
        scan_file = os.path.join(final_output_path, "scans", f"scan_{sample_name}.pkl")
        label_file = os.path.join(
            final_output_path, "labels", f"label_{sample_name}.pkl"
        )

        if os.path.exists(scan_file):
            with open(scan_file, "rb") as f:
                data["final_scan_data"] = pickle.load(f)

        if os.path.exists(label_file):
            with open(label_file, "rb") as f:
                data["final_label_data"] = pickle.load(f)

        return data

    def _create_sample_visualization(self, sample_name: str, data: Dict) -> plt.Figure:
        """Create comprehensive visualization for a sample"""

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            f"Complete Pipeline: {sample_name}", fontsize=16, fontweight="bold"
        )

        # Row 1: Original → Processed → Simulation
        # Original BUSI scan
        if "original_scan" in data:
            axes[0, 0].imshow(data["original_scan"], cmap="gray")
            axes[0, 0].set_title("Original BUSI Scan")
        else:
            axes[0, 0].text(
                0.5, 0.5, "No Original\nScan Available", ha="center", va="center"
            )
            axes[0, 0].set_title("Original BUSI Scan")
        axes[0, 0].axis("off")

        # Original BUSI mask
        if "original_mask" in data:
            axes[0, 1].imshow(data["original_mask"], cmap="gray")
            axes[0, 1].set_title("Original BUSI Mask")
        else:
            axes[0, 1].text(
                0.5, 0.5, "No Original\nMask Available", ha="center", va="center"
            )
            axes[0, 1].set_title("Original BUSI Mask")
        axes[0, 1].axis("off")

        # Sound speed map
        if "sound_speed_map" in data and data["sound_speed_map"] is not None:
            z_middle = data["sound_speed_map"].shape[2] // 2
            sos_slice = data["sound_speed_map"][:, :, z_middle]
            im = axes[0, 2].imshow(sos_slice, cmap="jet")
            axes[0, 2].set_title("Sound Speed Map [m/s]")
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        else:
            axes[0, 2].text(
                0.5, 0.5, "No Sound Speed\nMap Available", ha="center", va="center"
            )
            axes[0, 2].set_title("Sound Speed Map")
        axes[0, 2].axis("off")

        # Semantic tissue map
        if "semantic_map" in data and data["semantic_map"] is not None:
            z_middle = data["semantic_map"].shape[2] // 2
            semantic_slice = data["semantic_map"][:, :, z_middle]
            im = axes[0, 3].imshow(semantic_slice, cmap="viridis", vmin=1, vmax=4)
            axes[0, 3].set_title("Tissue Semantic Map")
            cbar = plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
            cbar.set_ticks([1, 2, 3, 4])
            cbar.set_ticklabels(["Bg", "Fat", "Gland", "Tumor"])
        else:
            axes[0, 3].text(
                0.5, 0.5, "No Semantic\nMap Available", ha="center", va="center"
            )
            axes[0, 3].set_title("Tissue Semantic Map")
        axes[0, 3].axis("off")

        # Row 2: Final processed results
        # B-mode ultrasound
        if "final_scan_data" in data and data["final_scan_data"] is not None:
            b_mode = data["final_scan_data"].get("noisy_us_scan_b_mode")
            if b_mode is not None:
                axes[1, 0].imshow(b_mode, cmap="gray")
                axes[1, 0].set_title("Synthetic B-mode US")
            else:
                axes[1, 0].text(
                    0.5, 0.5, "No B-mode\nData Available", ha="center", va="center"
                )
                axes[1, 0].set_title("Synthetic B-mode US")
        else:
            axes[1, 0].text(
                0.5, 0.5, "No B-mode\nData Available", ha="center", va="center"
            )
            axes[1, 0].set_title("Synthetic B-mode US")
        axes[1, 0].axis("off")

        # Harmonic ultrasound
        if "final_scan_data" in data and data["final_scan_data"] is not None:
            harmonic = data["final_scan_data"].get("noisy_us_scan_harmonic")
            if harmonic is not None:
                axes[1, 1].imshow(harmonic, cmap="gray")
                axes[1, 1].set_title("Synthetic Harmonic US")
            else:
                axes[1, 1].text(
                    0.5, 0.5, "No Harmonic\nData Available", ha="center", va="center"
                )
                axes[1, 1].set_title("Synthetic Harmonic US")
        else:
            axes[1, 1].text(
                0.5, 0.5, "No Harmonic\nData Available", ha="center", va="center"
            )
            axes[1, 1].set_title("Synthetic Harmonic US")
        axes[1, 1].axis("off")

        # Binary label
        if "final_label_data" in data and data["final_label_data"] is not None:
            binary_label = data["final_label_data"].get("clean_phantom_binary")
            if binary_label is not None:
                axes[1, 2].imshow(binary_label, cmap="gray")
                axes[1, 2].set_title("Binary Tumor Label")
            else:
                axes[1, 2].text(
                    0.5, 0.5, "No Binary\nLabel Available", ha="center", va="center"
                )
                axes[1, 2].set_title("Binary Tumor Label")
        else:
            axes[1, 2].text(
                0.5, 0.5, "No Binary\nLabel Available", ha="center", va="center"
            )
            axes[1, 2].set_title("Binary Tumor Label")
        axes[1, 2].axis("off")

        # Gray label
        if "final_label_data" in data and data["final_label_data"] is not None:
            gray_label = data["final_label_data"].get("clean_phantom_gray")
            if gray_label is not None:
                axes[1, 3].imshow(gray_label, cmap="gray")
                axes[1, 3].set_title("Gray Tumor Label")
            else:
                axes[1, 3].text(
                    0.5, 0.5, "No Gray\nLabel Available", ha="center", va="center"
                )
                axes[1, 3].set_title("Gray Tumor Label")
        else:
            axes[1, 3].text(
                0.5, 0.5, "No Gray\nLabel Available", ha="center", va="center"
            )
            axes[1, 3].set_title("Gray Tumor Label")
        axes[1, 3].axis("off")

        plt.tight_layout()
        return fig

    def _create_individual_component_images(self, sample_name: str, data: Dict) -> Dict:
        """Create individual component images for detailed logging"""

        images = {}
        figures_to_close = []  # Keep track of figures to close later

        # Individual component images with all key visualizations
        # Original BUSI data
        if "original_scan" in data and data["original_scan"] is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(data["original_scan"], cmap="gray")
            ax.set_title(f"Original BUSI Scan - {sample_name}")
            ax.axis("off")
            images[f"components/{sample_name}/original_scan"] = wandb.Image(
                fig, caption=f"Original BUSI scan for {sample_name}"
            )
            figures_to_close.append(fig)

        if "original_mask" in data and data["original_mask"] is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(data["original_mask"], cmap="gray")
            ax.set_title(f"Original BUSI Mask - {sample_name}")
            ax.axis("off")
            images[f"components/{sample_name}/original_mask"] = wandb.Image(
                fig, caption=f"Original BUSI mask for {sample_name}"
            )
            figures_to_close.append(fig)

        # Simulation results
        if "sound_speed_map" in data and data["sound_speed_map"] is not None:
            z_middle = data["sound_speed_map"].shape[2] // 2
            sos_slice = data["sound_speed_map"][:, :, z_middle]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sos_slice, cmap="jet")
            ax.set_title(f"Sound Speed Map - {sample_name}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Speed [m/s]")
            images[f"components/{sample_name}/sound_speed_map"] = wandb.Image(
                fig, caption=f"Sound speed map for {sample_name}"
            )
            figures_to_close.append(fig)

        if "semantic_map" in data and data["semantic_map"] is not None:
            z_middle = data["semantic_map"].shape[2] // 2
            semantic_slice = data["semantic_map"][:, :, z_middle]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(semantic_slice, cmap="viridis", vmin=1, vmax=4)
            ax.set_title(f"Tissue Semantic Map - {sample_name}")
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticks([1, 2, 3, 4])
            cbar.set_ticklabels(["Background", "Fatty", "Glandular", "Tumor"])
            images[f"components/{sample_name}/semantic_map"] = wandb.Image(
                fig, caption=f"Tissue semantic map for {sample_name}"
            )
            figures_to_close.append(fig)

        # Final processed ultrasound images
        if "final_scan_data" in data and data["final_scan_data"] is not None:
            b_mode = data["final_scan_data"].get("noisy_us_scan_b_mode")
            if b_mode is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(b_mode, cmap="gray")
                ax.set_title(f"Synthetic B-mode US - {sample_name}")
                ax.axis("off")
                images[f"components/{sample_name}/synthetic_bmode"] = wandb.Image(
                    fig, caption=f"Synthetic B-mode ultrasound for {sample_name}"
                )
                figures_to_close.append(fig)

            harmonic = data["final_scan_data"].get("noisy_us_scan_harmonic")
            if harmonic is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(harmonic, cmap="gray")
                ax.set_title(f"Synthetic Harmonic US - {sample_name}")
                ax.axis("off")
                images[f"components/{sample_name}/synthetic_harmonic"] = wandb.Image(
                    fig, caption=f"Synthetic harmonic ultrasound for {sample_name}"
                )
                figures_to_close.append(fig)

        # Final labels
        if "final_label_data" in data and data["final_label_data"] is not None:
            binary_label = data["final_label_data"].get("clean_phantom_binary")
            if binary_label is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(binary_label, cmap="gray")
                ax.set_title(f"Binary Tumor Label - {sample_name}")
                ax.axis("off")
                images[f"components/{sample_name}/binary_label"] = wandb.Image(
                    fig, caption=f"Binary tumor label for {sample_name}"
                )
                figures_to_close.append(fig)

            gray_label = data["final_label_data"].get("clean_phantom_gray")
            if gray_label is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(gray_label, cmap="gray")
                ax.set_title(f"Gray Tumor Label - {sample_name}")
                ax.axis("off")
                images[f"components/{sample_name}/gray_label"] = wandb.Image(
                    fig, caption=f"Gray tumor label for {sample_name}"
                )
                figures_to_close.append(fig)

        # Store figures to close later - add them to the return dict with a special key
        images["_figures_to_close"] = figures_to_close

        return images

    def _create_sample_config(
        self, sample_name: str, data: Dict, sample_index: int
    ) -> Dict:
        """Create sample-specific configuration data"""

        sample_metadata = self.sample_metadata.get(sample_name, {})

        # Use wandb.config.update() for configuration instead of wandb.log() to avoid media type conflicts
        config_data = {
            f"sample_info/{sample_name}/index": sample_index,
            f"sample_info/{sample_name}/category": sample_metadata.get(
                "category", "unknown"
            ),
            f"sample_info/{sample_name}/original_id": sample_metadata.get(
                "original_id", "unknown"
            ),
            f"sample_info/{sample_name}/original_filename": sample_metadata.get(
                "original_filename", "unknown"
            ),
            f"sample_info/{sample_name}/has_original_scan": "original_scan" in data,
            f"sample_info/{sample_name}/has_original_mask": "original_mask" in data,
            f"sample_info/{sample_name}/has_simulation_data": "scan_lines" in data,
            f"sample_info/{sample_name}/has_final_data": "final_scan_data" in data,
        }

        # Add dimensional information
        if "sound_speed_map" in data and data["sound_speed_map"] is not None:
            shape = data["sound_speed_map"].shape
            config_data[f"sample_info/{sample_name}/phantom_dimensions"] = (
                f"{shape[0]}x{shape[1]}x{shape[2]}"
            )

        if "scan_lines" in data and data["scan_lines"] is not None:
            shape = data["scan_lines"].shape
            config_data[f"sample_info/{sample_name}/scan_lines_shape"] = (
                f"{shape[0]}x{shape[1]}"
            )
            config_data[f"sample_info/{sample_name}/kgrid_Nt"] = data.get("kgrid_Nt", 0)
            config_data[f"sample_info/{sample_name}/kgrid_dt"] = data.get("kgrid_dt", 0)

        # Update wandb config instead of logging as metrics
        wandb.config.update(config_data)

        # Return empty dict since we're not logging these as metrics anymore
        return {}

    def finalize_wandb(self, results: Dict, total_time: float):
        """Finalize WandB run with summary statistics"""

        if not self.wandb_run:
            return

        # Log final summary
        summary_data = {
            "pipeline_summary/total_execution_time_minutes": total_time / 60,
            "pipeline_summary/samples_processed": len(
                results.get("post_processing", {})
            ),
            "pipeline_summary/simulation_success": len(results.get("simulation", {})),
            "pipeline_summary/post_processing_success": len(
                results.get("post_processing", {})
            ),
            "pipeline_summary/success_rate": len(results.get("post_processing", {}))
            / self.config.max_samples,
        }

        wandb.log(summary_data)

        # Create summary table
        if results.get("post_processing"):
            table_data = []

            # Extract actual sample names from post-processing keys
            for key in results["post_processing"].keys():
                if key.startswith("sample_") and key.endswith("_low"):
                    sample_name = key[7:-4]  # Remove "sample_" prefix and "_low" suffix
                    sample_metadata = self.sample_metadata.get(sample_name, {})
                    table_data.append(
                        [
                            sample_name,
                            sample_metadata.get("category", "unknown"),
                            sample_metadata.get("original_id", "unknown"),
                            (
                                "Success"
                                if sample_name in results.get("simulation", {})
                                else "Failed"
                            ),
                            (
                                "Success"
                                if key in results.get("post_processing", {})
                                else "Failed"
                            ),
                        ]
                    )

            table = wandb.Table(
                columns=[
                    "Sample Name",
                    "Category",
                    "Original ID",
                    "Simulation",
                    "Post-Processing",
                ],
                data=table_data,
            )
            wandb.log({"pipeline_summary/sample_results_table": table})

        print(f"WandB run finalized: {self.wandb_run.get_url()}")


class PipelineRunner:
    """Main pipeline runner class"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processed_sample_names = []
        self.wandb_logger = None

        # Create output directories
        os.makedirs(self.config.processed_phantoms_path, exist_ok=True)
        os.makedirs(self.config.simulation_results_path, exist_ok=True)
        os.makedirs(self.config.final_output_path, exist_ok=True)

    def run_complete_pipeline(self) -> Dict:
        """Run the complete pipeline"""

        print("=" * 60)
        print("Starting Python k-Wave Ultrasound Simulation Pipeline")
        print("=" * 60)

        start_time = time.time()
        results = {}

        try:
            # Initialize WandB if enabled
            if self.config.enable_wandb and WANDB_AVAILABLE:
                # Setup simulation config for WandB logger
                sim_config = SimulationConfig()
                if self.config.use_gpu:
                    sim_config.data_cast = "gpuArray-single"

                self.wandb_logger = WandBLogger(self.config, sim_config)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"pipeline_run_{timestamp}_{self.config.max_samples}samples"
                self.wandb_logger.initialize_wandb(run_name)
                self.wandb_logger.load_sample_metadata(
                    self.config.processed_phantoms_path
                )

            # Stage 1: Data Preparation
            if self.config.run_data_preparation:
                print("\n" + "=" * 40)
                print("STAGE 1: Data Preparation")
                print("=" * 40)
                preparation_results = self._run_data_preparation()
                results["data_preparation"] = preparation_results

                # Get available sample IDs
                if self.config.sample_names_to_process is None:
                    self.processed_sample_names = list(preparation_results.keys())
                else:
                    self.processed_sample_names = self.config.sample_names_to_process

                # Limit number of samples if specified
                if self.config.max_samples:
                    self.processed_sample_names = self.processed_sample_names[
                        : self.config.max_samples
                    ]
            else:
                # If skipping data preparation, load available sample names from metadata
                self._load_existing_sample_names()

            # Stage 2: k-Wave Simulation
            if self.config.run_simulation:
                print("\n" + "=" * 40)
                print("STAGE 2: k-Wave Simulation")
                print("=" * 40)
                simulation_results = self._run_simulation()
                results["simulation"] = simulation_results

            # Stage 3: Post-Processing
            if self.config.run_post_processing:
                print("\n" + "=" * 40)
                print("STAGE 3: Post-Processing")
                print("=" * 40)
                post_processing_results = self._run_post_processing()
                results["post_processing"] = post_processing_results

                # Log to WandB after post-processing (when all data is available)
                if self.wandb_logger:
                    self._log_samples_to_wandb(results)

            # Pipeline Summary
            total_time = time.time() - start_time
            self._print_pipeline_summary(results, total_time)

            # Save pipeline results
            self._save_pipeline_results(results)

            # Finalize WandB
            if self.wandb_logger:
                self.wandb_logger.finalize_wandb(results, total_time)

        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            raise

        return results

    def _log_samples_to_wandb(self, results: Dict):
        """Log all samples to WandB with comprehensive data - flexible for multiple noise levels"""
        if not self.wandb_logger:
            return

        print("\n" + "=" * 40)
        print("LOGGING TO WANDB")
        print("=" * 40)

        # Get successfully processed samples and extract actual sample names
        post_processing_results = results.get("post_processing", {})

        # Debug: print all keys to see what we're working with
        print(f"Post-processing result keys: {list(post_processing_results.keys())}")

        # Dynamically detect available noise levels and samples
        sample_noise_combinations = {}
        detected_noise_levels = set()

        for key in post_processing_results.keys():
            if key.startswith("sample_"):
                # Parse keys like "sample_benign_1_low" or "sample_malignant_5_medium"
                parts = key.split("_")
                if len(parts) >= 3:  # sample_category_id_noiselevel
                    noise_level = parts[-1]  # Last part is noise level
                    sample_name = "_".join(
                        parts[1:-1]
                    )  # Everything between 'sample_' and noise level

                    detected_noise_levels.add(noise_level)

                    if sample_name not in sample_noise_combinations:
                        sample_noise_combinations[sample_name] = []
                    sample_noise_combinations[sample_name].append(noise_level)

                    print(
                        f"Detected: sample='{sample_name}', noise='{noise_level}' from key='{key}'"
                    )

        print(f"Detected noise levels: {sorted(detected_noise_levels)}")
        print(f"Found {len(sample_noise_combinations)} unique samples")

        # Log each sample with all available noise levels
        for sample_idx, (sample_name, available_noise_levels) in enumerate(
            sample_noise_combinations.items()
        ):
            print(
                f"\nLogging sample {sample_idx+1}/{len(sample_noise_combinations)}: {sample_name}"
            )
            print(f"  Available noise levels: {available_noise_levels}")

            # Use the first available noise level, or prefer 'low' if available
            preferred_noise_order = ["low", "medium", "high"]
            noise_level_to_use = available_noise_levels[0]  # Default to first available

            for preferred in preferred_noise_order:
                if preferred in available_noise_levels:
                    noise_level_to_use = preferred
                    break

            print(f"  Using noise level: {noise_level_to_use}")

            try:
                self.wandb_logger.log_sample_experiment(
                    sample_name=sample_name,
                    sample_index=sample_idx,
                    processed_phantoms_path=self.config.processed_phantoms_path,
                    simulation_results_path=self.config.simulation_results_path,
                    final_output_path=self.config.final_output_path,
                    noise_level=noise_level_to_use,
                )

                # If multiple noise levels available, log comparison
                if len(available_noise_levels) > 1:
                    self._log_noise_level_comparison(
                        sample_name, available_noise_levels, sample_idx
                    )

            except Exception as e:
                print(f"  WARNING: Failed to log {sample_name}: {e}")
                continue

        print(f"\nAll {len(sample_noise_combinations)} samples logged to WandB")

    def _log_noise_level_comparison(
        self, sample_name: str, noise_levels: List[str], sample_idx: int
    ):
        """Log comparison between different noise levels for the same sample"""
        if not self.wandb_logger or not self.wandb_logger.wandb_run:
            return

        try:
            # Create noise level comparison visualization
            fig, axes = plt.subplots(
                len(noise_levels), 2, figsize=(12, 4 * len(noise_levels))
            )
            if len(noise_levels) == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f"Noise Level Comparison - {sample_name}", fontsize=14)

            for i, noise_level in enumerate(noise_levels):
                # Load data for this noise level
                data = self.wandb_logger._load_sample_data(
                    sample_name,
                    self.config.processed_phantoms_path,
                    self.config.simulation_results_path,
                    self.config.final_output_path,
                    noise_level,
                )

                # Plot B-mode image
                if "final_scan_data" in data and data["final_scan_data"]:
                    b_mode = data["final_scan_data"].get("noisy_us_scan_b_mode")
                    if b_mode is not None:
                        axes[i, 0].imshow(b_mode, cmap="gray")
                        axes[i, 0].set_title(f"B-mode - {noise_level}")
                        axes[i, 0].axis("off")

                # Plot harmonic image
                if "final_scan_data" in data and data["final_scan_data"]:
                    harmonic = data["final_scan_data"].get("noisy_us_scan_harmonic")
                    if harmonic is not None:
                        axes[i, 1].imshow(harmonic, cmap="gray")
                        axes[i, 1].set_title(f"Harmonic - {noise_level}")
                        axes[i, 1].axis("off")

            plt.tight_layout()

            # Log to WandB
            wandb.log(
                {
                    f"comparisons/{sample_name}/noise_levels": wandb.Image(
                        fig, caption=f"Noise level comparison for {sample_name}"
                    ),
                    f"comparisons/{sample_name}/available_noise_levels": noise_levels,
                    f"comparisons/{sample_name}/sample_index": sample_idx,
                }
            )

            # Small delay to ensure WandB has processed the images on Windows
            time.sleep(0.1)

            plt.close(fig)
            print(f"  Noise level comparison logged for {sample_name}")

        except Exception as e:
            print(f"  WARNING: Failed to log noise comparison for {sample_name}: {e}")

    def _run_data_preparation(self) -> Dict:
        """Run data preparation stage"""

        print(f"Processing BUSI dataset from: {self.config.busi_dataset_path}")
        print(f"Output will be saved to: {self.config.processed_phantoms_path}")

        processor = PhantomDataProcessor(target_size=self.config.target_phantom_size)

        processed_samples = processor.process_busi_dataset(
            base_folder=self.config.busi_dataset_path,
            output_folder=self.config.processed_phantoms_path,
            visualize=self.config.show_phantoms,
        )

        print(
            f"Data preparation completed: {len(processed_samples)} samples processed"
        )
        return processed_samples

    def _load_existing_sample_names(self):
        """Load existing sample names from metadata when skipping data preparation"""
        metadata_file = os.path.join(
            self.config.processed_phantoms_path, "sample_metadata.json"
        )

        if not os.path.exists(metadata_file):
            print(f"Warning: No existing sample metadata found at {metadata_file}")
            print("Please run data preparation first or provide sample names manually")
            self.processed_sample_names = []
            return

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Get available sample names
            if self.config.sample_names_to_process is None:
                self.processed_sample_names = list(metadata.keys())
            else:
                self.processed_sample_names = self.config.sample_names_to_process

            # Sort by category and then by ID for consistent ordering
            def sort_key(sample_name):
                parts = sample_name.split("_")
                if len(parts) == 2:
                    category, id_str = parts
                    try:
                        sample_id = int(id_str)
                        return (category, sample_id)
                    except ValueError:
                        return (category, 0)
                return (sample_name, 0)

            self.processed_sample_names.sort(key=sort_key)

            # Limit number of samples if specified
            if self.config.max_samples:
                self.processed_sample_names = self.processed_sample_names[
                    : self.config.max_samples
                ]

            print(f"Loaded {len(self.processed_sample_names)} existing sample names")
            if self.processed_sample_names:
                print(
                    f"Sample range: {self.processed_sample_names[0]} to {self.processed_sample_names[-1]}"
                )

        except Exception as e:
            print(f"Error loading existing sample names: {e}")
            self.processed_sample_names = []

    def _run_simulation(self) -> Dict:
        """Run k-Wave simulation stage"""

        if self.config.phantom_type == "bezier":
            return self._run_bezier_simulation()

        print(f"Running simulations for {len(self.processed_sample_names)} samples")
        print(f"Noise levels: {self.config.noise_levels}")
        print(f"Output path: {self.config.simulation_results_path}")

        # Setup simulation configuration
        sim_config = SimulationConfig()
        if self.config.use_gpu:
            sim_config.data_cast = "gpuArray-single"
            print("GPU acceleration enabled")
        else:
            sim_config.data_cast = "single"
            print("CPU simulation mode")

        simulator = UltrasoundSimulator(sim_config)

        simulation_results = {}

        for i, sample_name in enumerate(self.processed_sample_names):
            print(
                f"\nProcessing sample {sample_name} ({i+1}/{len(self.processed_sample_names)})"
            )

            try:
                sample_results = simulator.run_simulation(
                    sample_name=sample_name,
                    input_data_path=self.config.processed_phantoms_path,
                    output_path=self.config.simulation_results_path,
                    noise_levels=self.config.noise_levels,
                    run_simulation=True,
                )

                simulation_results[sample_name] = sample_results
                print(f"Sample {sample_name} simulation completed")

            except Exception as e:
                print(f"Error simulating sample {sample_name}: {e}")
                continue

        print(
            f"\nSimulation stage completed: {len(simulation_results)} samples successfully simulated"
        )
        return simulation_results

    def _run_bezier_simulation(self) -> Dict:
        """Run a single Bezier phantom simulation (small, local run)"""
        print("Running Bezier phantom simulation")
        print(f"Output path: {self.config.simulation_results_path}")

        sim_config = SimulationConfig()
        if self.config.use_gpu:
            sim_config.data_cast = "gpuArray-single"
            print("GPU acceleration enabled")
        else:
            sim_config.data_cast = "single"
            print("CPU simulation mode")

        if self.config.small_simulation:
            sim_config.Nx = 96
            sim_config.Ny = 96
            sim_config.Nz = 16
            sim_config.pml_x_size = 6
            sim_config.pml_y_size = 6
            sim_config.pml_z_size = 3
            sim_config.number_scan_lines = 16
            sim_config.element_width = 1
            sim_config.x_size = 20e-3

        simulator = UltrasoundSimulator(sim_config)

        bezier_params = {
            "radius": 0.003,
            "x_shift": 10,
            "y_shift": 10,
            "z_shift": 4,
            "background_map_mean": 1.0,
            "background_map_std": 0.008,
            "scattering_mean_c1": 25,
            "scattering_std_c2": 2.5,
            "scattering_divider_c3": 1.5,
        }

        result = simulator.run_simulation_bezier(
            phantom_name=self.config.bezier_name,
            output_path=self.config.simulation_results_path,
            bezier_params=bezier_params,
            run_simulation=True,
        )

        return {self.config.bezier_name: result}

    def _run_post_processing(self) -> Dict:
        """Run post-processing stage"""

        print(f"Post-processing {len(self.processed_sample_names)} samples")
        print(f"Output path: {self.config.final_output_path}")

        # Setup post-processing configuration
        post_config = PostProcessingConfig(
            target_shape=self.config.target_phantom_size,
            show_preview=self.config.show_post_processing_previews,
        )

        # Run batch post-processing
        results = process_batch(
            simulation_results_path=self.config.simulation_results_path,
            output_path=self.config.final_output_path,
            sample_names=self.processed_sample_names,
            noise_levels=self.config.noise_levels,
            config=post_config,
        )

        print(f"Post-processing completed: {len(results)} samples processed")
        return results

    def _print_pipeline_summary(self, results: Dict, total_time: float):
        """Print pipeline execution summary"""

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)

        print(
            f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )

        if "data_preparation" in results:
            print(
                f"Data preparation: {len(results['data_preparation'])} samples processed"
            )

        if "simulation" in results:
            print(f"Simulation: {len(results['simulation'])} samples simulated")

        if "post_processing" in results:
            print(
                f"Post-processing: {len(results['post_processing'])} samples processed"
            )

        print(f"\nFinal training data available at: {self.config.final_output_path}")

        # Check final output structure
        if os.path.exists(self.config.final_output_path):
            scans_path = os.path.join(self.config.final_output_path, "scans")
            labels_path = os.path.join(self.config.final_output_path, "labels")

            if os.path.exists(scans_path):
                scan_files = [f for f in os.listdir(scans_path) if f.endswith(".pkl")]
                print(f"Scan files created: {len(scan_files)}")

            if os.path.exists(labels_path):
                label_files = [f for f in os.listdir(labels_path) if f.endswith(".pkl")]
                print(f"Label files created: {len(label_files)}")

        print("Pipeline completed successfully!")

    def _save_pipeline_results(self, results: Dict):
        """Save pipeline results and configuration"""

        # Save configuration
        config_file = os.path.join(
            self.config.final_output_path, "pipeline_config.json"
        )
        with open(config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save execution summary
        summary = {
            "processed_sample_names": self.processed_sample_names,
            "noise_levels": self.config.noise_levels,
            "total_samples": len(self.processed_sample_names),
            "stages_completed": list(results.keys()),
            "output_paths": {
                "phantoms": self.config.processed_phantoms_path,
                "simulations": self.config.simulation_results_path,
                "training_data": self.config.final_output_path,
            },
        }

        summary_file = os.path.join(
            self.config.final_output_path, "execution_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Pipeline configuration saved to: {config_file}")
        print(f"Execution summary saved to: {summary_file}")


def load_config_from_file(config_path: str) -> PipelineConfig:
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return PipelineConfig.from_dict(config_dict)


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Run Python k-Wave Ultrasound Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run first 100 samples end-to-end (DEFAULT)
  python run_python_pipeline.py --end-sample-id 100 --gpu
  
  # Run specific sample range
  python run_python_pipeline.py --sample-range 1 50 --gpu
  
  # Run specific samples
  python run_python_pipeline.py --sample-names benign_10 malignant_5 --gpu
  
  # Run first 10 samples with multiple noise levels
  python run_python_pipeline.py --end-sample-id 10 --noise-levels low medium high --gpu
  
  # Run samples 50-100 simulation only
  python run_python_pipeline.py --sample-range 50 100 --stages sim --gpu
  
  # Run with custom WandB directory
  python run_python_pipeline.py --end-sample-id 50 --gpu --wandb-dir /path/to/wandb/logs
  
  # Run phantom analysis for sample range
  python run_python_pipeline.py --sample-range 1 20 --stages phantom-analysis
        """,
    )

    # Sample selection arguments
    sample_group = parser.add_mutually_exclusive_group(required=False)
    sample_group.add_argument(
        "--sample-names",
        type=str,
        nargs="+",
        help="Specific sample names to process (e.g., benign_10 malignant_5)",
    )
    sample_group.add_argument(
        "--sample-name",
        type=str,
        help="Single sample name to process (shorthand for --sample-names)",
    )
    sample_group.add_argument(
        "--sample-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Process samples by ID range (e.g., --sample-range 1 100)",
    )

    parser.add_argument(
        "--start-sample-id",
        type=int,
        default=1,
        help="Starting sample ID for range processing (default: 1)",
    )
    parser.add_argument(
        "--end-sample-id",
        type=int,
        default=100,
        help="Ending sample ID for range processing (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process from the range",
    )

    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (when not using specific names)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--visualize", action="store_true", help="Show visualizations during processing"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Stages to run: all, prep, sim, post, phantom-analysis, or comma-separated list",
    )
    parser.add_argument(
        "--phantom-type",
        type=str,
        default="busi",
        choices=["busi", "bezier"],
        help="Phantom source: 'busi' (mask-based) or 'bezier' (synthetic)",
    )
    parser.add_argument(
        "--bezier-name",
        type=str,
        default="bezier_minimal",
        help="Name for bezier phantom output folder",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a small grid for a quick run (bezier mode only)",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        default=True,
        help="Automatically continue through all stages without user input (default: True)",
    )
    parser.add_argument("--wandb-project", type=str, help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, help="WandB entity (username/team)")
    parser.add_argument("--wandb-dir", type=str, help="Directory to store WandB files")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="+",
        default=None,
        help="Noise levels to process (e.g., --noise-levels low medium high)",
    )
    parser.add_argument(
        "--noise-level", type=str, help="Single noise level to process (shorthand)"
    )

    args = parser.parse_args()

    # Handle special case: phantom analysis only
    if args.stages == "phantom-analysis":
        return run_phantom_analysis_only(args)

    # Create configuration using centralized system
    print("Creating configuration from arguments...")

    # Load base configuration from file if provided, otherwise use defaults
    if args.config and os.path.exists(args.config):
        print(f"Loading base configuration from: {args.config}")
        config = load_config_from_file(args.config)
    else:
        print("Using default configuration")
        config = PipelineConfig()

    # Update configuration from command line arguments
    config.update_from_args(args)

    # Set sample names to process attribute
    config.sample_names_to_process = None
    config.phantom_type = args.phantom_type
    config.bezier_name = args.bezier_name
    config.small_simulation = bool(args.small)

    # Handle specific sample selection (busi only)
    if config.phantom_type == "busi" and args.sample_names:
        config.sample_names_to_process = args.sample_names
        print(f"Will process specific samples: {config.sample_names_to_process}")
    elif config.phantom_type == "busi" and args.sample_name:
        config.sample_names_to_process = [args.sample_name]
        print(f"Will process single sample: {config.sample_names_to_process}")
    elif config.phantom_type == "busi" and args.sample_range:
        start_id, end_id = args.sample_range
        # Use sequential range selection (treats all samples as 1-780 sequence)
        config.sample_names_to_process = get_samples_by_sequential_range(
            config.processed_phantoms_path, start_id, end_id, args.max_samples
        )
        if config.sample_names_to_process:
            print(
                f"Will process sequential sample range {start_id}-{end_id}: {len(config.sample_names_to_process)} samples"
            )
            print(
                f"  Range: {config.sample_names_to_process[0]} to {config.sample_names_to_process[-1]}"
            )
        else:
            print(f"ERROR: No samples found in sequential range {start_id}-{end_id}")
            return None
    elif config.phantom_type == "busi":
        # Use sequential range for start/end sample ID parameters as default
        config.sample_names_to_process = get_samples_by_sequential_range(
            config.processed_phantoms_path,
            args.start_sample_id,
            args.end_sample_id,
            args.max_samples,
        )
        if config.sample_names_to_process:
            print(
                f"Will process sequential samples {args.start_sample_id}-{args.end_sample_id}: {len(config.sample_names_to_process)} samples"
            )
            print(
                f"  Range: {config.sample_names_to_process[0]} to {config.sample_names_to_process[-1]}"
            )
        else:
            print(
                f"ERROR: No samples found in sequential range {args.start_sample_id}-{args.end_sample_id}"
            )
            print(f"Will process up to {args.samples} samples (auto-detected)")
    else:
        print(f"Using bezier phantom: {config.bezier_name}")

    # Ensure auto-continue is enabled for batch processing
    if config.sample_names_to_process and len(config.sample_names_to_process) > 5:
        print("Auto-continue enabled for batch processing")
        config.enable_wandb = True  # Enable WandB for large batch runs

    # Override config with additional command line arguments
    if args.samples and not config.sample_names_to_process:
        config.max_samples = args.samples

    # Set stages to run
    if args.stages != "all":
        config.run_data_preparation = False
        config.run_simulation = False
        config.run_post_processing = False

        stages = [s.strip() for s in args.stages.split(",")]
        # if 'prep' in stages:
        #     config.run_data_preparation = True
        # Already have done the data preparation step
        if "sim" in stages:
            config.run_simulation = True
        if "post" in stages:
            config.run_post_processing = True

    # For bezier runs, disable data prep and post-processing by default
    if config.phantom_type == "bezier":
        config.run_data_preparation = False
        if args.stages == "all":
            config.run_post_processing = False

    # Print configuration summary
    print("\nPipeline Configuration Summary:")
    print("-" * 50)
    if config.sample_names_to_process:
        print(f"Specific samples to process: {config.sample_names_to_process}")
    else:
        print(f"Max samples to process: {config.max_samples}")
    print(f"Noise levels: {config.noise_levels}")
    print(f"Use GPU: {config.use_gpu}")
    print(f"Data cast: {config.data_cast}")
    print(f"Show visualizations: {config.show_phantoms}")
    print(f"Data preparation: {config.run_data_preparation}")
    print(f"Simulation: {config.run_simulation}")
    print(f"Post-processing: {config.run_post_processing}")
    print(f"Phantom type: {config.phantom_type}")
    print(f"WandB enabled: {config.enable_wandb}")
    if config.enable_wandb:
        print(f"WandB project: {config.wandb_project}")
        if config.wandb_entity:
            print(f"WandB entity: {config.wandb_entity}")
        if config.wandb_dir:
            print(f"WandB directory: {config.wandb_dir}")
    print("-" * 50)

    # Check WandB availability if enabled
    if config.enable_wandb and not WANDB_AVAILABLE:
        print(
            "Warning: WandB requested but not available. Install with: pip install wandb"
        )
        print("Continuing without WandB logging...")
        config.enable_wandb = False

    # Run pipeline
    runner = PipelineRunner(config)
    results = runner.run_complete_pipeline()

    return results


def run_phantom_analysis_only(args):
    """Run only phantom analysis for specified samples"""

    if not args.sample_names and not args.sample_name:
        print("ERROR: Must specify sample name(s) for phantom analysis")
        print(
            "Usage: python run_python_pipeline.py --sample-names benign_10 --stages phantom-analysis"
        )
        return None

    # Determine samples to analyze
    if args.sample_names:
        sample_names = args.sample_names
    elif args.sample_name:
        sample_names = [args.sample_name]

    print("=" * 60)
    print("🔬 PHANTOM ANALYSIS MODE")
    print("=" * 60)
    print(f"📋 Samples to analyze: {', '.join(sample_names)}")

    # Import the phantom analysis function
    from tests.run_phantom_analysis import run_phantom_analysis

    # Default input path
    input_data_path = (
        r"D:\CMME\1_data_generation\data\BUSI\processed_data_python_pipeline"
    )

    results = {}
    success_count = 0

    for sample_name in sample_names:
        print(f"\n🔍 Analyzing sample: {sample_name}")
        success = run_phantom_analysis(sample_name, input_data_path)
        results[sample_name] = success
        if success:
            success_count += 1

    print(
        f"\n- Phantom analysis completed: {success_count}/{len(sample_names)} successful"
    )

    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
