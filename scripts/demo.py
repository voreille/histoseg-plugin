#!/usr/bin/env python3
"""
Demo script for histoseg WSI processing

This script demonstrates how to use the histoseg-plugin preprocessing module
to process whole slide images.
"""

from pathlib import Path
import sys

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent / "src"))

from histoseg_plugin.tiling import process_single_wsi, process_wsi_directory


def demo_single_wsi():
    """Demonstrate processing a single WSI file"""
    print("=== Demo: Processing Single WSI ===")
    
    # Example paths (adjust as needed)
    wsi_path = Path("/home/val/data/DHMC_LUAD/test_corrected/DHMC_0010.tif")  # Replace with actual WSI file
    output_dir = Path("output/demo_single")
    
    if not wsi_path.exists():
        print(f"WSI file not found: {wsi_path}")
        print("Please provide a valid WSI file path")
        return
    
    result = process_single_wsi(
        wsi_path=wsi_path,
        output_dir=output_dir,
        patch_size=256,
        step_size=256,
        patch_level=0,
        generate_mask=True,
        generate_patches=True,
        generate_stitch=True,
        verbose=True
    )
    
    print(f"Result: {result}")
    print(f"Check outputs in: {output_dir}")


def demo_directory():
    """Demonstrate processing a directory of WSI files"""
    print("\\n=== Demo: Processing Directory ===")
    
    # Example paths (adjust as needed)
    source_dir = Path("/home/val/data/DHMC_LUAD/test_corrected")  # Replace with actual directory
    output_dir = Path("output/demo_batch")
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Please provide a valid directory with WSI files")
        return
    
    results = process_wsi_directory(
        source_dir=source_dir,
        output_dir=output_dir,
        patch_size=256,
        step_size=256,
        patch_level=0,
        generate_mask=True,
        generate_patches=True,
        generate_stitch=True,
        auto_skip=True,
        verbose=True
    )
    
    print(f"Processed {len(results)} files")
    for result in results:
        print(f"  {result['slide_id']}: {result['status']}")


def demo_cli_usage():
    """Show CLI usage examples"""
    print("\\n=== Demo: CLI Usage Examples ===")
    
    print("After installing the package with 'pip install -e .', you can use:")
    print()
    print("# Basic processing:")
    print("histoseg-process --source /path/to/wsi/files --output /path/to/results")
    print()
    print("# Custom patch size:")
    print("histoseg-process --source /data --output /results --patch-size 512")
    print()
    print("# Skip stitching for faster processing:")
    print("histoseg-process --source /data --output /results --no-stitch")
    print()
    print("# Process only patches (no masks or stitches):")
    print("histoseg-process --source /data --output /results --no-seg --no-stitch")
    print()
    print("# Quiet mode:")
    print("histoseg-process --source /data --output /results --quiet")


if __name__ == "__main__":
    print("Histoseg WSI Processing Demo")
    print("=" * 40)
    
    # demo_cli_usage()
    
    # Uncomment to test with actual data:
    demo_single_wsi()
    demo_directory()
    
    print("\\nDemo complete! Adjust paths in this script to test with your WSI data.")