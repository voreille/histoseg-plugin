# Test Summary: Configuration and MPP Selection

## Overview
Comprehensive test suite for the histoseg-plugin configuration system and MPP selection logic, covering all critical functionality with 44 test cases.

## Test Coverage

### ✅ MPP Extraction (3 tests)
- MPP metadata extraction from WSI properties
- Fallback to downsample ratios when metadata missing
- Graceful handling of corrupted metadata

### ✅ Fixed Mode Selection (6 tests)
- Basic level selection and clamping to valid ranges
- Optional tolerance checking with target MPP
- Single-level WSI handling

### ✅ Auto Mode Selection (6 tests)
- Policy-based selection: "closest", "lower", "higher"
- Tolerance checking with detailed error messages
- Fallback behavior when no levels match policy criteria

### ✅ Configuration Dispatcher (4 tests)
- Correct routing between fixed and auto modes
- Integration with Pydantic validation
- Error handling for invalid configurations

### ✅ Pydantic Model Validation (6 tests)
- Fixed mode validation (requires tile_level ≥ 0)
- Auto mode validation (requires target_tile_mpp)
- Parameter bounds checking (tile_size, mpp_tolerance)

### ✅ Configuration Overrides (9 tests)
- Mode switching (auto ↔ fixed) with explicit and implicit triggers
- Geometry-only updates preserving current mode
- Error handling for incomplete mode switches
- Convenience methods (to_fixed, to_auto)

### ✅ Configuration Loading (4 tests)
- YAML config loading with preset inheritance
- Deep merging of configuration sections
- Multiple preset application in order

### ✅ Integration Tests (3 tests)
- End-to-end pipeline testing for both modes
- Configuration override → MPP selection workflow
- Mode consistency validation

### ✅ Edge Cases (3 tests)
- WSI with no pyramid levels
- Zero tolerance boundary conditions
- Helper function validation

## Key Features Tested

### 🔧 **Dual-Mode Architecture**
- **Fixed Mode**: Use explicit tile_level with optional tolerance checking
- **Auto Mode**: Policy-based selection using target_tile_mpp and level_policy

### 📊 **Policy Implementation** 
- **"closest"**: Select level with MPP closest to target
- **"lower"**: Select highest resolution level ≤ target (fallback to closest)
- **"higher"**: Select lowest resolution level ≥ target (fallback to closest)

### ⚙️ **Configuration System**
- **Smart Mode Switching**: Automatic inference based on provided parameters
- **Preset Inheritance**: Deep merging of YAML configurations
- **Runtime Overrides**: Immutable config updates with validation

### 🛡️ **Error Handling**
- **Pydantic Validation**: Catch invalid configurations at creation time
- **Typed Exceptions**: Clear error messages for different failure modes
- **Boundary Conditions**: Graceful handling of edge cases

## Test Infrastructure

### Mock Objects
- `mock_wsi_standard`: 4-level pyramid with MPP metadata
- `mock_wsi_no_mpp`: 3-level pyramid without metadata  
- `mock_wsi_single_level`: Single-level WSI for edge testing

### Fixtures
- `sample_configs`: Pre-configured Config objects for different modes
- `temp_yaml_configs`: Temporary YAML files for loading tests

### Dependencies
- **pytest**: Test framework
- **pytest-mock**: Mocking support
- **unittest.mock**: WSI object mocking

## Usage

```bash
# Run all configuration tests
pytest tests/test_config_and_mpp_selection.py

# Run specific test class
pytest tests/test_config_and_mpp_selection.py::TestMPPSelectionAuto

# Run with coverage
pytest tests/test_config_and_mpp_selection.py --cov=histoseg_plugin.tiling
```

This test suite provides comprehensive coverage of the configuration and MPP selection system, ensuring reliable behavior across all supported use cases and edge conditions.