#!/usr/bin/env python3
"""
Enhanced WSI Processing Workflow with Parameter Management

This script demonstrates the improved parameter management system that replaces
CLAM's CSV-based approach with a flexible JSON-based system.
"""

from pathlib import Path
import sys

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent / "src"))

from histoseg_plugin.tiling import WSIParameterManager


def demo_parameter_workflow():
    """Demonstrate the enhanced parameter management workflow"""
    print("=== Enhanced Parameter Management Demo ===")
    
    # Initialize parameter manager
    config_dir = Path("config_demo")
    manager = WSIParameterManager(config_dir)
    
    # 1. Show available presets
    print("\\n1. Available presets:")
    presets = manager.list_presets()
    for preset in presets:
        print(f"   • {preset}")
    
    # 2. Process a WSI with default parameters
    slide_id = "example_slide_001"
    print(f"\\n2. Default parameters for {slide_id}:")
    default_params = manager.get_wsi_params(slide_id)
    print(f"   Segmentation threshold: {default_params.seg_params.sthresh}")
    print(f"   Area threshold: {default_params.filter_params.a_t}")
    print(f"   Patch size: {default_params.patch_size}")
    
    # 3. Simulate processing and finding issues
    print(f"\\n3. Simulating processing of {slide_id}...")
    print("   ❌ Issue found: Too much background captured")
    print("   💡 Solution: Increase segmentation threshold")
    
    # 4. Adjust parameters for this specific WSI
    print("\\n4. Setting WSI-specific parameter overrides:")
    manager.set_wsi_overrides(slide_id, {
        'seg_params': {'sthresh': 12},  # Increase threshold
        'filter_params': {'a_t': 150}   # Increase area threshold
    })
    
    # 5. Check new effective parameters
    updated_params = manager.get_wsi_params(slide_id)
    print(f"   New segmentation threshold: {updated_params.seg_params.sthresh}")
    print(f"   New area threshold: {updated_params.filter_params.a_t}")
    
    # 6. Record processing attempt
    print("\\n5. Recording processing attempt...")
    manager.record_processing(
        slide_id=slide_id,
        params=updated_params,
        status='success',
        processing_time=45.2,
        outputs={'mask': True, 'patches': True, 'stitch': True},
        notes='Parameters adjusted after initial run - good tissue coverage'
    )
    
    # 7. Show processing history
    print("\\n6. Processing history:")
    history = manager.get_processing_history(slide_id)
    for i, record in enumerate(history, 1):
        print(f"   Run {i}: {record.status} ({record.processing_time:.1f}s)")
        if record.notes:
            print(f"          Notes: {record.notes}")
    
    # 8. Create a preset from successful configuration
    print("\\n7. Creating preset from successful configuration...")
    manager.save_preset("high_background_slides", updated_params)
    print("   Preset 'high_background_slides' created")
    
    # 9. Process another similar slide using the new preset
    slide_id2 = "example_slide_002"
    print(f"\\n8. Processing {slide_id2} with new preset:")
    preset_params = manager.get_wsi_params(slide_id2, preset="high_background_slides")
    print(f"   Using preset parameters - sthresh: {preset_params.seg_params.sthresh}")
    
    print("\\n=== Workflow Complete ===")
    print(f"Configuration files saved in: {config_dir}")


def demo_cli_workflow():
    """Show CLI workflow examples"""
    print("\\n=== CLI Workflow Examples ===")
    
    print("""
After installing with 'pip install -e .', you get two powerful CLI tools:

🔧 WSI Processing:
   histoseg-process --source /path/to/slides --output /results

📊 Parameter Management:
   
   # View processing history
   histoseg-params history slide_001
   
   # Adjust parameters for problematic slide
   histoseg-params set-overrides slide_001 --sthresh 12 --a-t 150
   
   # Check current parameters
   histoseg-params show-params slide_001
   
   # Get adjustment workflow guide
   histoseg-params workflow slide_001
   
   # Create preset from working configuration
   histoseg-params create-preset good_tissue --from-wsi slide_001
   
   # Process new slides with preset
   histoseg-params show-params slide_002 --preset good_tissue

🎯 Benefits over CLAM's CSV approach:

   ✅ Individual JSON files per WSI (easier to manage)
   ✅ Parameter inheritance (global → preset → WSI-specific)
   ✅ Processing history tracking
   ✅ Easy parameter adjustment workflow
   ✅ Preset management for different slide types
   ✅ Visual feedback integration
   ✅ No more monolithic CSV files to manage
   ✅ Version control friendly
   """)


def demo_integration_example():
    """Show how to integrate with processing pipeline"""
    print("\\n=== Integration Example ===")
    
    print("""
# Integrated processing with parameter management:

from histoseg_plugin.tiling import WSIParameterManager, process_single_wsi

# Initialize parameter manager
manager = WSIParameterManager("config")

# Process WSI with smart parameter management
slide_id = "problematic_slide"
params = manager.get_wsi_params(slide_id, preset="biopsy")

# Check if reprocessing is needed
output_files = {
    'mask': Path(f"output/masks/{slide_id}.jpg"),
    'patches': Path(f"output/patches/{slide_id}.h5"),
    'stitch': Path(f"output/stitches/{slide_id}.jpg")
}

if manager.needs_reprocessing(slide_id, params, output_files):
    print(f"Processing {slide_id} with current parameters...")
    
    # Process the WSI (your existing function)
    result = process_single_wsi(
        wsi_path=f"data/{slide_id}.svs",
        output_dir="output",
        **params.to_dict()
    )
    
    # Record the processing attempt
    manager.record_processing(
        slide_id=slide_id,
        params=params,
        status=result['status'],
        processing_time=result.get('total_time', 0),
        outputs={
            'mask': Path(f"output/masks/{slide_id}.jpg").exists(),
            'patches': Path(f"output/patches/{slide_id}.h5").exists(), 
            'stitch': Path(f"output/stitches/{slide_id}.jpg").exists()
        }
    )
else:
    print(f"Skipping {slide_id} - outputs up to date")
    """)


if __name__ == "__main__":
    print("Enhanced WSI Parameter Management Demo")
    print("=" * 50)
    
    demo_parameter_workflow()
    demo_cli_workflow()
    demo_integration_example()
    
    print("\\n🎉 Demo complete!")
    print("\\nThis system provides much better parameter management than CLAM's CSV approach:")
    print("- Individual configuration per WSI")
    print("- Processing history tracking") 
    print("- Easy parameter adjustment workflow")
    print("- Preset management")
    print("- CLI tools for parameter manipulation")
    print("- Integration with visual feedback")