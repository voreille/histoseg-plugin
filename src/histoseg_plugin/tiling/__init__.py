"""
Tiling Module - WSI Preprocessing and Parameter Management

This module provides tools for tiling whole slide images, including:
- Tissue segmentation
- Patch coordinate extraction  
- Parameter management and tracking
- Visual feedback workflow
"""

from .process_wsi import process_single_wsi, process_wsi_directory
from .contour_proc import process_single_contour, process_contours_to_hdf5
from .parameter_manager import (
    TilingParams,
    SegmentationParams,
    FilterParams,
    VisualizationParams,
    PatchParams,
    WSIParameterManager,
    ProcessingRecord
)

__all__ = [
    # Processing functions
    'process_single_wsi',
    'process_wsi_directory',
    'process_single_contour',
    'process_contours_to_hdf5',
    # Parameter management
    'TilingParams',
    'SegmentationParams', 
    'FilterParams',
    'VisualizationParams',
    'PatchParams',
    'WSIParameterManager',
    'ProcessingRecord',
]