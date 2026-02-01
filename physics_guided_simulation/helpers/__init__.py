"""
Helper utilities for the Ultrasound Simulation Pipeline
=======================================================

This package contains utility functions for:
- File operations and path handling
- Data loading and saving
- Common helper functions
- Default configuration options
"""

from .file_utils import (
    get_available_samples,
    get_samples_by_id_range,
    load_json_safely,
    save_json_safely,
    ensure_directory_exists,
    get_file_size_mb
)

from .base_options import (
    BaseOptions,
    SimulationConfig,
    PostProcessingConfig,
    DataPreparationConfig,
    QuickCommands,
    DefaultPaths,
    default_options,
    default_paths,
    quick_commands,
    get_preset_config,
    list_available_presets,
    print_preset_info,
    create_config_from_args,
    PRESET_CONFIGS
)

__all__ = [
    # File utilities
    'get_available_samples',
    'get_samples_by_id_range', 
    'load_json_safely',
    'save_json_safely',
    'ensure_directory_exists',
    'get_file_size_mb',
    
    # Configuration classes
    'BaseOptions',
    'SimulationConfig',
    'PostProcessingConfig', 
    'DataPreparationConfig',
    'QuickCommands', 
    'DefaultPaths',
    
    # Pre-configured instances
    'default_options',
    'default_paths',
    'quick_commands',
    
    # Configuration management
    'get_preset_config',
    'list_available_presets',
    'print_preset_info',
    'create_config_from_args',
    'PRESET_CONFIGS'
] 