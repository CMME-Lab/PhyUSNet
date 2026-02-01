"""
File Utilities for Ultrasound Simulation Pipeline
=================================================

Common file operations and path handling utilities.
"""

import os
import json
from typing import List, Dict, Optional

def get_available_samples(input_data_path: str) -> List[str]:
    """Get list of available sample names from the input data path"""
    
    if not os.path.exists(input_data_path):
        print(f"ERROR: Input data path does not exist: {input_data_path}")
        return []
    
    # Find all subdirectories that contain mask.png
    sample_names = []
    for item in os.listdir(input_data_path):
        sample_path = os.path.join(input_data_path, item)
        if os.path.isdir(sample_path):
            mask_path = os.path.join(sample_path, "mask.png")
            if os.path.exists(mask_path):
                sample_names.append(item)
    
    return sorted(sample_names)

def get_samples_by_sequential_range(input_data_path: str, start_index: int, end_index: int, max_samples: int = None) -> List[str]:
    """
    Get sample names by sequential range (1-780) regardless of internal category structure.
    
    Args:
        input_data_path: Path to processed phantoms
        start_index: Starting sequential index (1-based, e.g., 600)
        end_index: Ending sequential index (1-based, e.g., 647) 
        max_samples: Optional limit on number of samples
        
    Returns:
        List of sample names corresponding to the sequential range
    """
    if not os.path.exists(input_data_path):
        print(f"ERROR: Input data path does not exist: {input_data_path}")
        return []
    
    # Get all available samples in sorted order
    all_samples = get_available_samples(input_data_path)
    
    # Sort samples by category and ID for consistent ordering
    def sort_key(sample_name):
        parts = sample_name.split('_')
        if len(parts) == 2:
            category, id_str = parts
            try:
                sample_id = int(id_str)
                return (category, sample_id)
            except ValueError:
                return (category, 0)
        return (sample_name, 0)
    
    all_samples.sort(key=sort_key)
    
    total_samples = len(all_samples)
    print(f"Total available samples: {total_samples}")
    print(f"Requested sequential range: {start_index}-{end_index}")
    
    # Validate range
    if start_index < 1:
        print(f"WARNING: Start index {start_index} is too low, using 1")
        start_index = 1
        
    if end_index > total_samples:
        print(f"WARNING: End index {end_index} is too high, using {total_samples}")
        end_index = total_samples
        
    if start_index > end_index:
        print(f"ERROR: Invalid range: start {start_index} > end {end_index}")
        return []
    
    # Convert to 0-based indexing and extract the range
    start_idx = start_index - 1  # Convert to 0-based
    end_idx = min(end_index, total_samples)  # Convert to 0-based, inclusive
    
    selected_samples = all_samples[start_idx:end_idx]
    
    # Apply max_samples limit if specified
    if max_samples and max_samples < len(selected_samples):
        selected_samples = selected_samples[:max_samples]
    
    print(f"- Selected {len(selected_samples)} samples from sequential positions {start_index}-{min(end_index, total_samples)}")
    if selected_samples:
        print(f"   Range: {selected_samples[0]} to {selected_samples[-1]}")
    
    return selected_samples

def get_samples_by_id_range(input_data_path: str, start_id: int, end_id: int, max_samples: int = None) -> List[str]:
    """Get sample names by ID range from the input data path (legacy category-based method)"""
    
    if not os.path.exists(input_data_path):
        print(f"ERROR: Input data path does not exist: {input_data_path}")
        return []
    
    # Find all available samples
    all_samples = get_available_samples(input_data_path)
    
    # Sort samples by category and ID
    def sort_key(sample_name):
        parts = sample_name.split('_')
        if len(parts) == 2:
            category, id_str = parts
            try:
                sample_id = int(id_str)
                return (category, sample_id)
            except ValueError:
                return (category, 0)
        return (sample_name, 0)
    
    all_samples.sort(key=sort_key)
    
    # Filter by ID range
    filtered_samples = []
    for sample_name in all_samples:
        parts = sample_name.split('_')
        if len(parts) == 2:
            category, id_str = parts
            try:
                sample_id = int(id_str)
                if start_id <= sample_id <= end_id:
                    filtered_samples.append(sample_name)
            except ValueError:
                continue
    
    # Apply max_samples limit if specified
    if max_samples and max_samples < len(filtered_samples):
        filtered_samples = filtered_samples[:max_samples]
    
    return filtered_samples

def load_json_safely(file_path: str) -> Optional[Dict]:
    """Safely load JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON from {file_path}: {e}")
        return None

def save_json_safely(data: Dict, file_path: str) -> bool:
    """Safely save data to JSON file with error handling"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error: Could not save JSON to {file_path}: {e}")
        return False

def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error: Could not create directory {directory_path}: {e}")
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0.0 