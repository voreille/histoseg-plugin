"""
CLI for WSI processing - Extract patches, masks, and visualizations from whole slide images.

This module provides a command-line interface for processing WSI files using the
process_wsi module. It replicates the functionality of CLAM's create_patches_fp.py
but as a clean, modular implementation for the histoseg plugin.
"""

from pathlib import Path

import click

from .process_wsi import process_wsi_directory


@click.command()
@click.option(
    "--source", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing WSI files"
)
@click.option(
    "--output", 
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for processed data (will create masks/, patches/, stitches/ subdirs)"
)
@click.option(
    "--patch-size", 
    type=int, 
    default=256, 
    help="Size of patches to extract (default: 256)"
)
@click.option(
    "--step-size", 
    type=int, 
    default=256, 
    help="Step size for patch extraction (default: 256)"
)
@click.option(
    "--patch-level", 
    type=int, 
    default=0, 
    help="Pyramid level for patch extraction (default: 0)"
)
@click.option(
    "--seg/--no-seg", 
    default=True, 
    help="Generate segmentation masks (default: enabled)"
)
@click.option(
    "--patch/--no-patch", 
    default=True, 
    help="Generate patch coordinates (default: enabled)"
)
@click.option(
    "--stitch/--no-stitch", 
    default=True, 
    help="Generate stitched visualizations (default: enabled)"
)
@click.option(
    "--auto-skip/--no-auto-skip", 
    default=True, 
    help="Skip processing if outputs already exist (default: enabled)"
)
@click.option(
    "--quiet", 
    is_flag=True, 
    help="Suppress verbose output"
)
@click.option(
    "--extensions",
    default=".svs,.ndpi,.tiff,.tif",
    help="Comma-separated list of valid file extensions (default: .svs,.ndpi,.tiff,.tif)"
)
@click.option(
    "--preset",
    default="default",
    help="Name of the preset configuration to use (default: default) you can find available presets in config/presets/"
)
def main(
    source: Path, 
    output: Path, 
    patch_size: int, 
    step_size: int, 
    patch_level: int,
    seg: bool,
    patch: bool, 
    stitch: bool,
    auto_skip: bool,
    quiet: bool,
    extensions: str,
    preset: str,
):
    """
    Process WSI files to generate segmentation masks, patch coordinates, and stitched visualizations.
    
    This command processes all WSI files in the SOURCE directory and creates:
    - masks/: Binary segmentation masks (.jpg files)
    - patches/: Patch coordinate files (.h5 files) 
    - stitches/: Stitched visualizations for debugging (.jpg files)
    
    Examples:
    
    \b
    # Basic processing with default parameters
    histoseg-process --source /path/to/wsi/files --output /path/to/results
    
    \b
    # Process with custom patch size and skip stitching
    histoseg-process --source /data/slides --output /results --patch-size 512 --no-stitch
    
    \b
    # Process only patches (no masks or stitches)
    histoseg-process --source /data --output /results --no-seg --no-stitch
    """
    
    # Parse extensions
    file_extensions = tuple(ext.strip() for ext in extensions.split(','))
    
    if not quiet:
        click.echo(f"Processing WSI files from: {source}")
        click.echo(f"Output directory: {output}")
        click.echo(f"Patch size: {patch_size}, Step size: {step_size}, Level: {patch_level}")
        click.echo(f"Generate masks: {seg}, patches: {patch}, stitches: {stitch}")
        click.echo(f"File extensions: {file_extensions}")
        click.echo(f"Using preset: {preset}")
        click.echo()
    
    try:
        results = process_wsi_directory(
            source_dir=source,
            output_dir=output,
            patch_size=patch_size,
            step_size=step_size,
            patch_level=patch_level,
            generate_mask=seg,
            generate_patches=patch,
            generate_stitch=stitch,
            auto_skip=auto_skip,
            file_extensions=file_extensions,
            verbose=not quiet,
        )
        
        # Print summary
        if not quiet:
            status_counts = {}
            for result in results:
                status = result['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            click.echo("\n" + "="*50)
            click.echo("PROCESSING COMPLETE")
            click.echo("="*50)
            for status, count in status_counts.items():
                click.echo(f"{status}: {count}")
            
            if results:
                processed_count = status_counts.get('processed', 0)
                total_count = len(results)
                success_rate = (processed_count / total_count) * 100 if total_count > 0 else 0
                click.echo(f"Success rate: {success_rate:.1f}% ({processed_count}/{total_count})")
        
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
