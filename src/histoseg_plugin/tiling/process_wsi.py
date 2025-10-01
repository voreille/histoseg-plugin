"""
WSI Processing Module - Core functionality for histopathology slide preprocessing

Provides functions to segment tissue, extract patch coordinates, and generate visualizations
from whole slide images. Designed as a clean, modular API for the histoseg pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
from tqdm import tqdm

# Add resources to path for CLAM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "resources" / "CLAM"))

from histoseg_plugin.tiling.WholeSlideImage import WholeSlideImage
from histoseg_plugin.tiling.wsi_utils import StitchCoords


def process_single_wsi(
    wsi_path: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 256,
    step_size: int = 256,
    patch_level: int = 0,
    seg_params: Optional[Dict] = None,
    filter_params: Optional[Dict] = None,
    vis_params: Optional[Dict] = None,
    patch_params: Optional[Dict] = None,
    generate_mask: bool = True,
    generate_patches: bool = True,
    generate_stitch: bool = True,
    auto_skip: bool = True,
    verbose: bool = True,
) -> Dict[str, Union[float, str, bool]]:
    """
    Process a single WSI file to generate segmentation mask, patch coordinates, and stitched visualization.
    
    Args:
        wsi_path: Path to the WSI file
        output_dir: Directory where outputs will be saved (masks/, patches/, stitches/ subdirs)
        patch_size: Size of patches to extract (default: 256)
        step_size: Step size for patch extraction (default: 256) 
        patch_level: Pyramid level for patch extraction (default: 0)
        seg_params: Segmentation parameters
        filter_params: Filtering parameters for contours
        vis_params: Visualization parameters
        patch_params: Patch extraction parameters
        generate_mask: Whether to generate segmentation mask (default: True)
        generate_patches: Whether to generate patch coordinates (default: True)
        generate_stitch: Whether to generate stitched visualization (default: True)
        auto_skip: Skip processing if outputs already exist (default: True)
        verbose: Print progress information (default: True)
        
    Returns:
        Dictionary with processing results including timing and status
    """
    # Set default parameters
    if seg_params is None:
        seg_params = {
            'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 
            'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'
        }
    
    if filter_params is None:
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    
    if vis_params is None:
        vis_params = {'vis_level': -1, 'line_thickness': 500}
    
    if patch_params is None:
        patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
    
    # Convert paths
    wsi_path = Path(wsi_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    masks_dir = output_dir / "masks"
    patches_dir = output_dir / "patches" 
    stitches_dir = output_dir / "stitches"
    
    for dir_path in [masks_dir, patches_dir, stitches_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get slide name without extension
    slide_id = wsi_path.stem
    
    # Check if outputs already exist
    mask_path = masks_dir / f"{slide_id}.jpg"
    patch_path = patches_dir / f"{slide_id}.h5"
    stitch_path = stitches_dir / f"{slide_id}.jpg"
    
    if auto_skip:
        outputs_exist = (
            (not generate_mask or mask_path.exists()) and
            (not generate_patches or patch_path.exists()) and
            (not generate_stitch or stitch_path.exists())
        )
        if outputs_exist:
            if verbose:
                print(f"Outputs for {slide_id} already exist, skipping...")
            return {
                'slide_id': slide_id,
                'status': 'skipped',
                'seg_time': 0,
                'patch_time': 0, 
                'stitch_time': 0
            }
    
    if verbose:
        print(f"Processing {slide_id}...")
    
    # Initialize WSI object
    try:
        wsi_object = WholeSlideImage(str(wsi_path))
    except Exception as e:
        print(f"Error loading {wsi_path}: {e}")
        return {
            'slide_id': slide_id,
            'status': 'failed_load',
            'error': str(e),
            'seg_time': 0,
            'patch_time': 0,
            'stitch_time': 0
        }
    
    # Auto-determine levels if needed
    current_seg_params = seg_params.copy()
    current_vis_params = vis_params.copy()
    
    if current_seg_params['seg_level'] < 0:
        if len(wsi_object.level_dim) == 1:
            current_seg_params['seg_level'] = 0
        else:
            wsi = wsi_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_seg_params['seg_level'] = best_level
    
    if current_vis_params['vis_level'] < 0:
        if len(wsi_object.level_dim) == 1:
            current_vis_params['vis_level'] = 0
        else:
            wsi = wsi_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_vis_params['vis_level'] = best_level
    
    # Handle keep_ids and exclude_ids
    for param_name in ['keep_ids', 'exclude_ids']:
        ids_str = str(current_seg_params[param_name])
        if ids_str != 'none' and len(ids_str) > 0:
            current_seg_params[param_name] = np.array(ids_str.split(',')).astype(int)
        else:
            current_seg_params[param_name] = []
    
    # Check if segmentation level is reasonable
    w, h = wsi_object.level_dim[current_seg_params['seg_level']]
    if w * h > 1e8:
        print(f'Level dimensions {w} x {h} too large for segmentation, aborting')
        return {
            'slide_id': slide_id,
            'status': 'failed_seg_size',
            'seg_time': 0,
            'patch_time': 0,
            'stitch_time': 0
        }
    
    # Timing variables
    seg_time = patch_time = stitch_time = 0
    
    # Segmentation
    if generate_mask or generate_patches or generate_stitch:
        if verbose:
            print("  Segmenting tissue...")
        start_time = time.time()
        try:
            wsi_object.segmentTissue(**current_seg_params, filter_params=filter_params)
            seg_time = time.time() - start_time
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return {
                'slide_id': slide_id,
                'status': 'failed_seg',
                'error': str(e),
                'seg_time': 0,
                'patch_time': 0,
                'stitch_time': 0
            }
    
    # Generate mask
    if generate_mask:
        if verbose:
            print("  Generating mask...")
        try:
            mask = wsi_object.visWSI(**current_vis_params)
            mask.save(str(mask_path))
        except Exception as e:
            print(f"Error generating mask: {e}")
    
    # Generate patches
    if generate_patches:
        if verbose:
            print("  Extracting patch coordinates...")
        start_time = time.time()
        try:
            current_patch_params = patch_params.copy()
            current_patch_params.update({
                'patch_level': patch_level,
                'patch_size': patch_size,
                'step_size': step_size,
                'save_path': str(patches_dir)
            })
            _ = wsi_object.process_contours(**current_patch_params)
            patch_time = time.time() - start_time
        except Exception as e:
            print(f"Error during patching: {e}")
            patch_time = 0
    
    # Generate stitch
    if generate_stitch and generate_patches and patch_path.exists():
        if verbose:
            print("  Generating stitched visualization...")
        start_time = time.time()
        try:
            heatmap = StitchCoords(str(patch_path), wsi_object, downscale=64, 
                                 bg_color=(0,0,0), alpha=-1, draw_grid=False)
            heatmap.save(str(stitch_path))
            stitch_time = time.time() - start_time
        except Exception as e:
            print(f"Error during stitching: {e}")
            stitch_time = 0
    
    if verbose:
        print(f"  Completed {slide_id} - Seg: {seg_time:.2f}s, Patch: {patch_time:.2f}s, Stitch: {stitch_time:.2f}s")
    
    return {
        'slide_id': slide_id,
        'status': 'processed',
        'seg_time': seg_time,
        'patch_time': patch_time,
        'stitch_time': stitch_time
    }


def process_wsi_directory(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 256,
    step_size: int = 256,
    patch_level: int = 0,
    seg_params: Optional[Dict] = None,
    filter_params: Optional[Dict] = None,
    vis_params: Optional[Dict] = None,
    patch_params: Optional[Dict] = None,
    generate_mask: bool = True,
    generate_patches: bool = True,
    generate_stitch: bool = True,
    auto_skip: bool = True,
    file_extensions: Tuple[str, ...] = ('.svs', '.ndpi', '.tiff', '.tif'),
    verbose: bool = True,
) -> List[Dict]:
    """
    Process all WSI files in a directory.
    
    Args:
        source_dir: Directory containing WSI files
        output_dir: Directory where outputs will be saved
        patch_size: Size of patches to extract (default: 256)
        step_size: Step size for patch extraction (default: 256)
        patch_level: Pyramid level for patch extraction (default: 0)
        seg_params: Segmentation parameters
        filter_params: Filtering parameters for contours
        vis_params: Visualization parameters
        patch_params: Patch extraction parameters
        generate_mask: Whether to generate segmentation masks (default: True)
        generate_patches: Whether to generate patch coordinates (default: True)
        generate_stitch: Whether to generate stitched visualizations (default: True)
        auto_skip: Skip processing if outputs already exist (default: True)
        file_extensions: Valid WSI file extensions (default: ('.svs', '.ndpi', '.tiff', '.tif'))
        verbose: Print progress information (default: True)
        
    Returns:
        List of processing results for each WSI file
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Find all WSI files
    wsi_files = []
    for ext in file_extensions:
        wsi_files.extend(source_dir.glob(f"*{ext}"))
        wsi_files.extend(source_dir.glob(f"*{ext.upper()}"))
    
    wsi_files = sorted(wsi_files)
    
    if not wsi_files:
        print(f"No WSI files found in {source_dir} with extensions {file_extensions}")
        return []
    
    if verbose:
        print(f"Found {len(wsi_files)} WSI files to process")
    
    # Process each file
    results = []
    total_seg_time = total_patch_time = total_stitch_time = 0
    
    for wsi_path in tqdm(wsi_files, desc="Processing WSI files"):
        result = process_single_wsi(
            wsi_path=wsi_path,
            output_dir=output_dir,
            patch_size=patch_size,
            step_size=step_size,
            patch_level=patch_level,
            seg_params=seg_params,
            filter_params=filter_params,
            vis_params=vis_params,
            patch_params=patch_params,
            generate_mask=generate_mask,
            generate_patches=generate_patches,
            generate_stitch=generate_stitch,
            auto_skip=auto_skip,
            verbose=verbose,
        )
        
        results.append(result)
        
        if result['status'] == 'processed':
            total_seg_time += float(result['seg_time'])
            total_patch_time += float(result['patch_time'])
            total_stitch_time += float(result['stitch_time'])
    
    # Print summary
    if verbose:
        processed_count = sum(1 for r in results if r['status'] == 'processed')
        if processed_count > 0:
            print("\nProcessing Summary:")
            print(f"  Processed: {processed_count}/{len(wsi_files)} files")
            print(f"  Average segmentation time: {total_seg_time/processed_count:.2f}s")
            print(f"  Average patching time: {total_patch_time/processed_count:.2f}s")
            print(f"  Average stitching time: {total_stitch_time/processed_count:.2f}s")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process WSI files for histopathology analysis")
    parser.add_argument("--source", type=str, required=True,
                       help="Path to directory containing WSI files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--patch_size", type=int, default=256,
                       help="Patch size (default: 256)")
    parser.add_argument("--step_size", type=int, default=256,
                       help="Step size for patch extraction (default: 256)")
    parser.add_argument("--patch_level", type=int, default=0,
                       help="Pyramid level for patch extraction (default: 0)")
    parser.add_argument("--no_mask", action="store_true",
                       help="Skip mask generation")
    parser.add_argument("--no_patches", action="store_true", 
                       help="Skip patch coordinate generation")
    parser.add_argument("--no_stitch", action="store_true",
                       help="Skip stitch visualization generation")
    parser.add_argument("--no_auto_skip", action="store_true",
                       help="Don't skip files with existing outputs")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = process_wsi_directory(
        source_dir=args.source,
        output_dir=args.output,
        patch_size=args.patch_size,
        step_size=args.step_size,
        patch_level=args.patch_level,
        generate_mask=not args.no_mask,
        generate_patches=not args.no_patches,
        generate_stitch=not args.no_stitch,
        auto_skip=not args.no_auto_skip,
        verbose=not args.quiet,
    )
    
    # Print final summary
    if not args.quiet:
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nFinal Results:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")