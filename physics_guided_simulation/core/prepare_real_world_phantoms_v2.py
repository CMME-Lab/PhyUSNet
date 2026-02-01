# %% [markdown]
# # Prepare Real-World Phantoms - Version 2
# 
# Improved version that preserves original sample IDs and follows cleaner structure
# 

# %%
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class PhantomDataProcessor:
    """Class to process real-world phantom data while preserving original IDs"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        
    def convert_pixel_class(self, mask: np.ndarray) -> np.ndarray:
        """Convert pixel classes according to template"""
        template_classes = {
            255: 100,  # Tumor/lesion region
            0: 255,    # Background
        }
        converted_mask = np.zeros_like(mask)
        for original_val, new_val in template_classes.items():
            converted_mask[mask == original_val] = new_val
        return converted_mask
    
    def preprocess_mask(self, mask_path: str, convert_classes: bool = True) -> np.ndarray:
        """Preprocess mask with optional pixel class conversion"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        if convert_classes:
            mask = self.convert_pixel_class(mask)
        return mask
    
    def process_busi_dataset(self, base_folder: str, output_folder: str, 
                           visualize: bool = False) -> Dict[str, Dict]:
        """Process BUSI dataset while preserving original IDs and filenames"""
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image and mask paths with original IDs
        image_mask_pairs = self._load_busi_paths_with_ids(base_folder)
        
        processed_samples = {}
        
        for original_filename, category, original_id, image_path, mask_paths in image_mask_pairs:
            print(f"Processing {category} sample {original_id} (original: {original_filename})...")
            
            # Load and resize scan
            scan = cv2.imread(image_path)
            scan = cv2.resize(scan, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Process mask
            mask_path = mask_paths[0] if isinstance(mask_paths, list) else mask_paths
            mask = self.preprocess_mask(mask_path, convert_classes=True)
            
            # Create sample folder with preserved original filename structure
            # Format: category_originalID (e.g., benign_1, malignant_15, normal_5)
            sample_name = f"{category}_{original_id}"
            sample_folder = os.path.join(output_folder, sample_name)
            os.makedirs(sample_folder, exist_ok=True)
            
            # Save scan and mask
            scan_path = os.path.join(sample_folder, "scan.png")
            mask_path_out = os.path.join(sample_folder, "mask.png")
            
            cv2.imwrite(scan_path, scan)
            cv2.imwrite(mask_path_out, mask)
            
            # Store metadata with original filename information
            processed_samples[sample_name] = {
                'original_filename': original_filename,
                'category': category,
                'original_id': original_id,
                'original_image_path': image_path,
                'original_mask_path': mask_path,
                'processed_folder': sample_folder,
                'scan_path': scan_path,
                'mask_path': mask_path_out
            }
            
            # Optional visualization
            if visualize:
                self._visualize_sample(scan, mask, sample_name, category)
            
            print(f"Saved sample {sample_name} to: {sample_folder}")
        
        # Save metadata with detailed mapping
        import json
        metadata_path = os.path.join(output_folder, "sample_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(processed_samples, f, indent=2)
        
        # Also save a simple mapping file for easy reference
        mapping_file = os.path.join(output_folder, "original_to_processed_mapping.txt")
        with open(mapping_file, 'w') as f:
            f.write("# Mapping from original BUSI filenames to processed sample names\n")
            f.write("# Format: processed_name -> original_filename (category)\n\n")
            for sample_name, info in processed_samples.items():
                f.write(f"{sample_name} -> {info['original_filename']} ({info['category']})\n")
        
        print(f"Processing complete! Processed {len(processed_samples)} samples")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Mapping file saved to: {mapping_file}")
        
        return processed_samples
    
    def _load_busi_paths_with_ids(self, base_folder: str) -> List[Tuple]:
        """Load BUSI paths while preserving original IDs and filenames"""
        image_mask_pairs = []
        
        for category in os.listdir(base_folder):
            category_path = os.path.join(base_folder, category)
            if not os.path.isdir(category_path):
                continue
                
            image_files = os.listdir(category_path)
            # Extract unique IDs from filenames
            image_ids = list(set([
                x.split("(")[1].split(")")[0] 
                for x in image_files 
                if "(" in x and ")" in x and not "_mask" in x
            ]))
            
            for img_id in image_ids:
                original_id = int(img_id)  # Keep as int for sorting
                original_filename = f"{category} ({img_id})"  # Store original filename base
                image_path = os.path.join(category_path, f"{category} ({img_id}).png")
                mask_path = os.path.join(category_path, f"{category} ({img_id})_mask.png")
                
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    image_mask_pairs.append((
                        original_filename, category, original_id, image_path, mask_path
                    ))
        
        # Sort by category first, then by original ID to maintain logical order
        image_mask_pairs.sort(key=lambda x: (x[1], x[2]))  # Sort by category, then ID
        return image_mask_pairs
    
    def _visualize_sample(self, scan: np.ndarray, mask: np.ndarray, 
                         sample_name: str, category: str):
        """Visualize a processed sample"""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(scan)
        plt.title(f"{category} - {sample_name} - Scan")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"{category} - {sample_name} - Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# %% Main execution for BUSI dataset
if __name__ == "__main__":
    # Configuration
    BUSI_BASE_FOLDER = r"D:\CMME\2_denoising_ultrasound_experiments\data_busi\Dataset_BUSI_with_GT"
    BUSI_OUTPUT_FOLDER = r"D:\CMME\1_data_generation\data\BUSI\processed_data_v2"
    
    # Process BUSI dataset
    processor = PhantomDataProcessor(target_size=(256, 256))
    
    try:
        processed_samples = processor.process_busi_dataset(
            base_folder=BUSI_BASE_FOLDER,
            output_folder=BUSI_OUTPUT_FOLDER,
            visualize=False  # Set to True to see preview of each sample
        )
        
        # Print summary
        categories = {}
        for sample_name, info in processed_samples.items():
            cat = info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nDataset Summary:")
        for category, count in categories.items():
            print(f"  {category}: {count} samples")
        
        # Print some example mappings
        print("\nExample mappings (first 5):")
        for i, (sample_name, info) in enumerate(list(processed_samples.items())[:5]):
            print(f"  {sample_name} -> {info['original_filename']}")
            
    except Exception as e:
        print(f"Error processing BUSI dataset: {e}")
        print("Please check the base folder path and ensure the dataset is available") 