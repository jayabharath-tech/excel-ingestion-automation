"""Unit tests for South African ID validation and field correction."""

import pandas as pd
import pytest

from core.constant import Gender
from core.data_cleaner import (
    validate_sa_id,
    extract_dob,
    extract_gender,
    correct_sa_id_fields
)


class TestSAIDValidation:
    """Test cases for South African ID validation functions."""

    def test_validate_sa_id_valid_ids(self):
        """Test validation of valid SA ID numbers."""
        # Valid SA ID numbers (pass Luhn checksum)
        valid_ids = pd.Series([
            "7612026128186",  # Valid
            "8001015009087",  # Valid
        ])
        
        result = validate_sa_id(valid_ids)
        
        assert result.all(), "All valid IDs should pass validation"

    def test_validate_sa_id_invalid_ids(self):
        """Test validation of invalid SA ID numbers."""
        # Invalid SA ID numbers (fail Luhn checksum or format)
        invalid_ids = pd.Series([
            "7630256128186",  # Invalid Luhn
            "123",            # Too short
            "abcdefghijklmnop",  # Non-numeric
            "",               # Empty
            "1234567890123",  # Invalid checksum
        ])
        
        result = validate_sa_id(invalid_ids)
        
        assert not result.any(), "All invalid IDs should fail validation"
        assert result.sum() == 0, "Should have 0 valid IDs"

    def test_validate_sa_id_mixed_valid_invalid(self):
        """Test validation with mixed valid and invalid IDs."""
        mixed_ids = pd.Series([
            "7612026128186",  # Valid
            "7630256128186",  # Invalid
            "8001015009087",  # Valid
            "123",            # Invalid
        ])
        
        result = validate_sa_id(mixed_ids)
        
        expected = [True, False, True, False]
        assert list(result) == expected
        assert result.sum() == 2, "Should have 2 valid IDs"

    def test_extract_dob_valid_ids(self):
        """Test DOB extraction from valid SA IDs."""
        ids = pd.Series([
            "7612026128186",  # 1976-12-02
            "8001015009087",  # 1980-01-01
            "0501015009087",  # 2005-01-01
        ])
        
        result = extract_dob(ids)
        
        # Check extracted dates
        assert pd.notna(result).all(), "All should have valid DOB"
        
        # Check specific dates
        assert result.iloc[0].year == 1976
        assert result.iloc[0].month == 12
        assert result.iloc[0].day == 2
        
        assert result.iloc[1].year == 1980
        assert result.iloc[1].month == 1
        assert result.iloc[1].day == 1
        
        assert result.iloc[2].year == 2005
        assert result.iloc[2].month == 1
        assert result.iloc[2].day == 1

    def test_extract_dob_invalid_dates(self):
        """Test DOB extraction with invalid dates."""
        ids = pd.Series([
            "7602305009087",  # Feb 30 (invalid)
            "7613025009087",  # Month 13 (invalid)
            "7600005009087",  # Day 00 (invalid)
        ])
        
        result = extract_dob(ids)
        
        # All should be NaT (Not a Time) due to invalid dates
        assert result.isna().all(), "All invalid dates should be NaT"

    def test_extract_gender(self):
        """Test gender extraction from SA IDs."""
        ids = pd.Series([
            "7612026128186",  # 6128 > 4999 -> Male
            "8001015009087",  # 5009 > 4999 -> Male  
            "9001014009087",  # 4009 <= 4999 -> Female
            "0501013009087",  # 3009 <= 4999 -> Female
        ])
        
        result = extract_gender(ids)
        
        expected = [Gender.MALE, Gender.MALE, Gender.FEMALE, Gender.FEMALE]
        assert list(result) == expected

    def test_correct_sa_id_fields_full_correction(self):
        """Test full correction when ID is valid and DOB is valid."""
        df = pd.DataFrame({
            "IdNo": ["7612026128186"],
            "Dob": [pd.to_datetime("01/01/1980")],  # Wrong DOB as datetime
            "Gender": [Gender.FEMALE]    # Wrong gender
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Should be corrected based on ID - check datetime format
        assert result_df.loc[0, "Dob"].year == 1976
        assert result_df.loc[0, "Dob"].month == 12
        assert result_df.loc[0, "Dob"].day == 2
        assert result_df.loc[0, "Gender"] == Gender.MALE

    def test_correct_sa_id_fields_invalid_id(self):
        """Test no correction when ID is invalid."""
        original_dob = "01/01/1980"
        original_gender = "Female"
        
        df = pd.DataFrame({
            "IdNo": ["7630256128186"],  # Invalid ID
            "Dob": [original_dob],
            "Gender": [original_gender]
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Should remain unchanged
        assert result_df.loc[0, "Dob"] == original_dob
        assert result_df.loc[0, "Gender"] == original_gender

    def test_correct_sa_id_fields_invalid_dob(self):
        """Test no correction when ID is valid but DOB extraction fails."""
        df = pd.DataFrame({
            "IdNo": ["7602305009087"],  # Valid ID but invalid date (Feb 30)
            "Dob": ["01/01/1980"],
            "Gender": ["Female"]
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Should remain unchanged due to invalid DOB
        assert result_df.loc[0, "Dob"] == "01/01/1980"
        assert result_df.loc[0, "Gender"] == "Female"

    def test_correct_sa_id_fields_mixed_records(self):
        """Test correction with mixed valid/invalid records."""
        df = pd.DataFrame({
            "IdNo": [
                "7612026128186",  # Valid ID + valid DOB
                "7630256128186",  # Invalid ID
                "7602305009087",  # Valid ID + invalid DOB
                "8001015009087"   # Valid ID + valid DOB
            ],
            "Dob": [
                pd.to_datetime( "01/01/1980"),     # Should be corrected
                pd.to_datetime( "01/01/1980"),     # Should remain unchanged
                pd.to_datetime( "01/01/1980"),     # Should remain unchanged
                pd.to_datetime( "01/01/1980" )     # Should be corrected
            ],
            "Gender": [
                "Female",         # Should be corrected to Male
                "Female",         # Should remain unchanged
                "Female",         # Should remain unchanged
                "Female"          # Should be corrected to Male
            ]
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Check which records were corrected
        # Record 0: Valid ID + valid DOB -> Should be corrected
        assert result_df.loc[0, "Dob"].year == 1976
        assert result_df.loc[0, "Gender"] == "Male"
        
        # Record 1: Invalid ID -> Should remain unchanged
        assert result_df.loc[1, "Dob"] == pd.to_datetime("01/01/1980")
        assert result_df.loc[1, "Gender"] == "Female"
        
        # Record 2: Valid ID + invalid DOB -> Should remain unchanged
        assert result_df.loc[2, "Dob"] == pd.to_datetime("01/01/1980")
        assert result_df.loc[2, "Gender"] == "Female"
        
        # Record 3: Valid ID + valid DOB -> Should be corrected
        assert result_df.loc[3, "Dob"].year == 1980
        assert result_df.loc[3, "Gender"] == "Male"

    def test_correct_sa_id_fields_empty_dataframe(self):
        """Test function with empty DataFrame."""
        df = pd.DataFrame({
            "IdNo": [],
            "Dob": [],
            "Gender": []
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Should return unchanged empty DataFrame
        assert len(result_df) == 0
        assert list(result_df.columns) == ["IdNo", "Dob", "Gender"]

    def test_correct_sa_id_fields_missing_columns(self):
        """Test function when required columns are missing."""
        df = pd.DataFrame({
            "FirstName": ["John"],
            "LastName": ["Doe"]
        })
        
        with pytest.raises(KeyError):
            correct_sa_id_fields(df, "IdNo", "Dob", "Gender")


    def test_correct_sa_id_fields_none_values(self):
        """Test function with None values in ID column."""
        df = pd.DataFrame({
            "IdNo": [None, "7612026128186", ""],
            "Dob": [pd.to_datetime("01/01/1980"), pd.to_datetime("01/01/1980"), pd.to_datetime("01/01/1980")],
            "Gender": ["Female", "Female", "Female"]
        })
        
        result_df = correct_sa_id_fields(df, "IdNo", "Dob", "Gender")
        
        # Only valid ID (index 1) should be corrected
        assert result_df.loc[0, "Dob"] == pd.to_datetime("01/01/1980")  # None ID - unchanged
        assert result_df.loc[0, "Gender"] == "Female"
        
        assert result_df.loc[1, "Dob"].year == 1976     # Valid ID - corrected
        assert result_df.loc[1, "Gender"] == "Male"
        
        assert result_df.loc[2, "Dob"] == pd.to_datetime("01/01/1980")  # Empty ID - unchanged
        assert result_df.loc[2, "Gender"] == "Female"


if __name__ == "__main__":
    pytest.main([__file__])
