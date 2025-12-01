"""
Property-based tests for negative tau_y support.

These tests validate the correctness properties
for the negative-tau-y-support feature.
"""

import pytest
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xspec_interface import validate_compps_params, COMPPS_LIMITS

# Import the validate_tau_values function from the script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_tau_kTe_study import (
    validate_tau_values,
    format_tau_for_filename,
    parse_tau_from_filename
)


class TestTauYValidationRange:
    """
    **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
    
    *For any* `tau_y` value, the `validate_compps_params` function should return 
    `(True, [])` if the value is in [-4, 3], and `(False, [error])` otherwise.
    
    **Validates: Requirements 1.3, 1.4**
    """

    @given(tau_y=st.floats(min_value=-4.0, max_value=3.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_valid_tau_y_values_accepted(self, tau_y: float):
        """
        Property: Any tau_y value within [-4, 3] should be accepted.
        
        **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
        **Validates: Requirements 1.3, 1.4**
        """
        params = {'tau_y': tau_y}
        is_valid, errors = validate_compps_params(params)
        
        assert is_valid, f"tau_y={tau_y} should be valid but got errors: {errors}"
        assert errors == [], f"tau_y={tau_y} should have no errors but got: {errors}"

    @given(tau_y=st.floats(max_value=-4.0 - 0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_tau_y_below_range_rejected(self, tau_y: float):
        """
        Property: Any tau_y value below -4 should be rejected.
        
        **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
        **Validates: Requirements 1.3, 1.4**
        """
        params = {'tau_y': tau_y}
        is_valid, errors = validate_compps_params(params)
        
        assert not is_valid, f"tau_y={tau_y} should be invalid (below -4)"
        assert len(errors) == 1, f"Expected exactly one error for tau_y={tau_y}"
        assert 'tau_y' in errors[0], f"Error should mention tau_y: {errors}"

    @given(tau_y=st.floats(min_value=3.0 + 0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_tau_y_above_range_rejected(self, tau_y: float):
        """
        Property: Any tau_y value above 3 should be rejected.
        
        **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
        **Validates: Requirements 1.3, 1.4**
        """
        params = {'tau_y': tau_y}
        is_valid, errors = validate_compps_params(params)
        
        assert not is_valid, f"tau_y={tau_y} should be invalid (above 3)"
        assert len(errors) == 1, f"Expected exactly one error for tau_y={tau_y}"
        assert 'tau_y' in errors[0], f"Error should mention tau_y: {errors}"

    def test_boundary_values(self):
        """
        Test exact boundary values for tau_y range.
        
        **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
        **Validates: Requirements 1.3, 1.4**
        """
        # Test lower boundary (-4.0) - should be valid
        is_valid, errors = validate_compps_params({'tau_y': -4.0})
        assert is_valid, f"tau_y=-4.0 (lower boundary) should be valid but got: {errors}"
        
        # Test upper boundary (3.0) - should be valid
        is_valid, errors = validate_compps_params({'tau_y': 3.0})
        assert is_valid, f"tau_y=3.0 (upper boundary) should be valid but got: {errors}"
        
        # Test typical negative y-parameter values
        for tau_y in [-0.5, -1.0, -2.0, -3.0]:
            is_valid, errors = validate_compps_params({'tau_y': tau_y})
            assert is_valid, f"tau_y={tau_y} should be valid but got: {errors}"
        
        # Test typical positive optical depth values
        for tau_y in [0.2, 0.5, 1.0, 1.5, 2.0]:
            is_valid, errors = validate_compps_params({'tau_y': tau_y})
            assert is_valid, f"tau_y={tau_y} should be valid but got: {errors}"

    def test_compps_limits_updated(self):
        """
        Verify that COMPPS_LIMITS has been updated for tau_y.
        
        **Feature: negative-tau-y-support, Property 1: tau_y validation handles extended range correctly**
        **Validates: Requirements 1.3, 1.4**
        """
        assert 'tau_y' in COMPPS_LIMITS, "tau_y should be in COMPPS_LIMITS"
        min_val, max_val = COMPPS_LIMITS['tau_y']
        assert min_val == -4.0, f"tau_y min should be -4.0, got {min_val}"
        assert max_val == 3.0, f"tau_y max should be 3.0, got {max_val}"


class TestTauValuesMixingValidation:
    """
    **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
    
    *For any* list of tau values, the `validate_tau_values` function should raise 
    `ValueError` if the list contains both positive and negative values, and return 
    the appropriate mode ('positive' or 'negative') otherwise.
    
    **Validates: Requirements 2.1, 2.2**
    """

    @given(tau_values=st.lists(
        st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10
    ))
    @settings(max_examples=100)
    def test_all_positive_values_return_positive_mode(self, tau_values: list):
        """
        Property: Any list of all positive tau values should return 'positive' mode.
        
        **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
        **Validates: Requirements 2.1, 2.2**
        """
        import logging
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        result = validate_tau_values(tau_values, logger)
        assert result == 'positive', f"All positive values {tau_values} should return 'positive', got '{result}'"

    @given(tau_values=st.lists(
        st.floats(min_value=-4.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10
    ))
    @settings(max_examples=100)
    def test_all_negative_values_return_negative_mode(self, tau_values: list):
        """
        Property: Any list of all negative tau values should return 'negative' mode.
        
        **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
        **Validates: Requirements 2.1, 2.2**
        """
        import logging
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        result = validate_tau_values(tau_values, logger)
        assert result == 'negative', f"All negative values {tau_values} should return 'negative', got '{result}'"

    @given(
        positive_values=st.lists(
            st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        ),
        negative_values=st.lists(
            st.floats(min_value=-4.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100)
    def test_mixed_values_raise_error(self, positive_values: list, negative_values: list):
        """
        Property: Any list containing both positive and negative values should raise ValueError.
        
        **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
        **Validates: Requirements 2.1, 2.2**
        """
        import logging
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        mixed_values = positive_values + negative_values
        
        with pytest.raises(ValueError) as exc_info:
            validate_tau_values(mixed_values, logger)
        
        assert "all positive or all negative" in str(exc_info.value).lower(), \
            f"Error message should mention 'all positive or all negative': {exc_info.value}"

    def test_empty_list_raises_error(self):
        """
        Test that empty list raises ValueError.
        
        **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
        **Validates: Requirements 2.1, 2.2**
        """
        import logging
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        with pytest.raises(ValueError) as exc_info:
            validate_tau_values([], logger)
        
        assert "empty" in str(exc_info.value).lower()

    def test_zero_value_raises_error(self):
        """
        Test that zero values raise ValueError.
        
        **Feature: negative-tau-y-support, Property 2: Mixed tau values are rejected, homogeneous accepted**
        **Validates: Requirements 2.1, 2.2**
        """
        import logging
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        with pytest.raises(ValueError) as exc_info:
            validate_tau_values([0.0], logger)
        
        assert "zero" in str(exc_info.value).lower()
        
        # Also test zero mixed with other values
        with pytest.raises(ValueError):
            validate_tau_values([0.5, 0.0, 1.0], logger)


class TestScenarioNameRoundTrip:
    """
    **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
    
    *For any* negative `tau_y` value, generating a scenario name (using 'm' prefix) 
    and then parsing it back should recover the original negative value.
    
    **Validates: Requirements 2.3, 4.2**
    """

    @given(tau=st.floats(min_value=-4.0, max_value=-0.01, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_negative_tau_round_trip(self, tau: float):
        """
        Property: For any negative tau value, format then parse should recover the original value.
        
        **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
        **Validates: Requirements 2.3, 4.2**
        """
        # Format the tau value for filename
        formatted = format_tau_for_filename(tau)
        
        # Verify it uses 'm' prefix for negative values
        assert formatted.startswith('m'), f"Negative tau {tau} should format with 'm' prefix, got '{formatted}'"
        
        # Parse it back
        parsed = parse_tau_from_filename(formatted)
        
        # Should recover the original value (within floating point precision of 2 decimal places)
        assert abs(parsed - round(tau, 2)) < 0.001, \
            f"Round-trip failed: {tau} -> '{formatted}' -> {parsed}"

    @given(tau=st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_positive_tau_round_trip(self, tau: float):
        """
        Property: For any positive tau value, format then parse should recover the original value.
        
        **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
        **Validates: Requirements 2.3, 4.2**
        """
        # Format the tau value for filename
        formatted = format_tau_for_filename(tau)
        
        # Verify it does NOT use 'm' prefix for positive values
        assert not formatted.startswith('m'), f"Positive tau {tau} should not have 'm' prefix, got '{formatted}'"
        
        # Parse it back
        parsed = parse_tau_from_filename(formatted)
        
        # Should recover the original value (within floating point precision of 2 decimal places)
        assert abs(parsed - round(tau, 2)) < 0.001, \
            f"Round-trip failed: {tau} -> '{formatted}' -> {parsed}"

    def test_specific_negative_values(self):
        """
        Test specific negative tau values for round-trip consistency.
        
        **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
        **Validates: Requirements 2.3, 4.2**
        """
        test_values = [-0.50, -1.00, -2.00, -3.00, -0.25, -1.75]
        
        for tau in test_values:
            formatted = format_tau_for_filename(tau)
            parsed = parse_tau_from_filename(formatted)
            
            # Check format
            expected_format = f"m{abs(tau):.2f}"
            assert formatted == expected_format, \
                f"tau={tau} should format as '{expected_format}', got '{formatted}'"
            
            # Check round-trip
            assert parsed == tau, \
                f"Round-trip failed for tau={tau}: got {parsed}"

    def test_format_examples(self):
        """
        Test specific format examples from requirements.
        
        **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
        **Validates: Requirements 2.3, 4.2**
        """
        # From requirements: -0.50 → "taum0.50"
        assert format_tau_for_filename(-0.50) == "m0.50"
        assert format_tau_for_filename(-1.00) == "m1.00"
        assert format_tau_for_filename(-2.50) == "m2.50"
        
        # Positive values should not have prefix
        assert format_tau_for_filename(0.50) == "0.50"
        assert format_tau_for_filename(1.00) == "1.00"
        assert format_tau_for_filename(2.50) == "2.50"

    def test_parse_examples(self):
        """
        Test specific parse examples.
        
        **Feature: negative-tau-y-support, Property 3: Scenario name round-trip for negative tau**
        **Validates: Requirements 2.3, 4.2**
        """
        # Parse negative format
        assert parse_tau_from_filename("m0.50") == -0.50
        assert parse_tau_from_filename("m1.00") == -1.00
        assert parse_tau_from_filename("m2.50") == -2.50
        
        # Parse positive format
        assert parse_tau_from_filename("0.50") == 0.50
        assert parse_tau_from_filename("1.00") == 1.00
        assert parse_tau_from_filename("2.50") == 2.50


class TestCSVOutputNegativeTau:
    """
    Tests for CSV output with negative tau values.
    
    Verifies that save_results_csv() correctly writes negative tau values
    as negative numbers in the CSV file.
    
    **Validates: Requirements 4.1**
    """

    def test_save_results_csv_with_negative_tau(self, tmp_path):
        """
        Test that save_results_csv() correctly writes negative tau values.
        
        **Validates: Requirements 4.1**
        """
        import logging
        from run_tau_kTe_study import save_results_csv
        
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        # Create test data with negative tau values
        results_by_kTe = {
            50.0: {
                'tau': [-0.50, -1.00, -2.00],
                'gamma': [2.5, 2.0, 1.5],
                'gamma_err_neg': [0.1, 0.1, 0.1],
                'gamma_err_pos': [0.1, 0.1, 0.1]
            },
            100.0: {
                'tau': [-0.50, -1.00, -2.00],
                'gamma': [2.3, 1.8, 1.3],
                'gamma_err_neg': [0.05, 0.05, 0.05],
                'gamma_err_pos': [0.05, 0.05, 0.05]
            }
        }
        
        # Save to temporary CSV file
        csv_file = tmp_path / "test_results.csv"
        save_results_csv(results_by_kTe, csv_file, logger)
        
        # Read the CSV file and verify negative values are preserved
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Check header
        assert lines[0].strip() == "kTe,tau_y,gamma,gamma_err_neg,gamma_err_pos"
        
        # Check that negative tau values are written as negative numbers
        # Expected lines for kTe=50.0
        assert "50.0,-0.5,2.5,0.1,0.1" in lines[1]
        assert "50.0,-1.0,2.0,0.1,0.1" in lines[2]
        assert "50.0,-2.0,1.5,0.1,0.1" in lines[3]
        
        # Expected lines for kTe=100.0
        assert "100.0,-0.5,2.3,0.05,0.05" in lines[4]
        assert "100.0,-1.0,1.8,0.05,0.05" in lines[5]
        assert "100.0,-2.0,1.3,0.05,0.05" in lines[6]

    def test_save_results_csv_with_positive_tau(self, tmp_path):
        """
        Test that save_results_csv() correctly writes positive tau values (baseline).
        
        **Validates: Requirements 4.1**
        """
        import logging
        from run_tau_kTe_study import save_results_csv
        
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        # Create test data with positive tau values
        results_by_kTe = {
            50.0: {
                'tau': [0.50, 1.00, 2.00],
                'gamma': [2.5, 2.0, 1.5],
                'gamma_err_neg': [0.1, 0.1, 0.1],
                'gamma_err_pos': [0.1, 0.1, 0.1]
            }
        }
        
        # Save to temporary CSV file
        csv_file = tmp_path / "test_results_positive.csv"
        save_results_csv(results_by_kTe, csv_file, logger)
        
        # Read the CSV file and verify positive values are preserved
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Check that positive tau values are written correctly
        assert "50.0,0.5,2.5,0.1,0.1" in lines[1]
        assert "50.0,1.0,2.0,0.1,0.1" in lines[2]
        assert "50.0,2.0,1.5,0.1,0.1" in lines[3]

    def test_save_results_csv_mixed_kTe_negative_tau(self, tmp_path):
        """
        Test CSV output with multiple kTe values and negative tau values.
        
        **Validates: Requirements 4.1**
        """
        import logging
        from run_tau_kTe_study import save_results_csv
        
        logger = logging.getLogger('test')
        logger.addHandler(logging.NullHandler())
        
        # Create test data with multiple kTe and negative tau values
        results_by_kTe = {
            50.0: {
                'tau': [-0.20, -0.50],
                'gamma': [3.0, 2.5],
                'gamma_err_neg': [0.15, 0.10],
                'gamma_err_pos': [0.20, 0.12]
            },
            100.0: {
                'tau': [-0.20, -0.50],
                'gamma': [2.5, 2.0],
                'gamma_err_neg': [0.08, 0.05],
                'gamma_err_pos': [0.10, 0.06]
            },
            150.0: {
                'tau': [-0.20, -0.50],
                'gamma': [2.0, 1.7],
                'gamma_err_neg': [0.05, 0.03],
                'gamma_err_pos': [0.06, 0.04]
            }
        }
        
        # Save to temporary CSV file
        csv_file = tmp_path / "test_results_mixed.csv"
        save_results_csv(results_by_kTe, csv_file, logger)
        
        # Read and parse the CSV file
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Verify we have the correct number of rows
        assert len(rows) == 6, f"Expected 6 rows (3 kTe × 2 tau), got {len(rows)}"
        
        # Verify all tau values are negative
        for row in rows:
            tau_value = float(row['tau_y'])
            assert tau_value < 0, f"Expected negative tau_y, got {tau_value}"
        
        # Verify specific values
        # kTe=50.0, tau=-0.20
        row_50_020 = [r for r in rows if float(r['kTe']) == 50.0 and abs(float(r['tau_y']) - (-0.20)) < 0.01][0]
        assert abs(float(row_50_020['gamma']) - 3.0) < 0.01
        
        # kTe=150.0, tau=-0.50
        row_150_050 = [r for r in rows if float(r['kTe']) == 150.0 and abs(float(r['tau_y']) - (-0.50)) < 0.01][0]
        assert abs(float(row_150_050['gamma']) - 1.7) < 0.01
