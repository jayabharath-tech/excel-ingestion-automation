import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from core.target_schema import repair_sa_id_using_dob


class TestRepairSAIdUsingDOB:
    """Test cases for repair_sa_id_using_dob function."""

    def test_repair_11_digit_id_born_2000(self):
        """Test repairing 11-digit ID for person born in 2000."""
        df = pd.DataFrame({
            "IdNo": ["01026128186"],  # 11 digits, first 4=0102 (Jan 2)
            "Dob": ["02/01/2000"]  # dd/mm/yyyy format: Jan 2, 2000, MMDD=0102
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should be padded with "00" for 2000 birth year
        assert result_df.loc[0, "IdNo"] == "0001026128186"

    def test_repair_12_digit_id_born_2001(self):
        """Test repairing 12-digit ID for person born in 2001."""
        df = pd.DataFrame({
            "IdNo": ["102266128186"],  # 12 digits, first 5=10226 (Feb 26, 2001)
            "Dob": ["26/02/2001"]  # Born Feb 26, 2001, YMMDD=10226
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should be padded with "0" for 2001-2009 birth years
        assert result_df.loc[0, "IdNo"] == "0102266128186"

    def test_repair_12_digit_id_born_2005(self):
        """Test repairing 12-digit ID for person born in 2005."""
        df = pd.DataFrame({
            "IdNo": ["501026128186"],  # 12 digits, missing "0"
            "Dob": ["02/01/2005"]  # Born 2005
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should be padded with "0" for 2001-2009 birth years
        assert result_df.loc[0, "IdNo"] == "0501026128186"

    def test_no_repair_13_digit_id(self):
        """Test that 13-digit IDs are not modified."""
        original_id = "7612026128186"
        df = pd.DataFrame({
            "IdNo": [original_id],  # 13 digits - correct length
            "Dob": ["02/12/1976"]
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should remain unchanged
        assert result_df.loc[0, "IdNo"] == original_id

    def test_no_repair_11_digit_id_not_born_2000(self):
        """Test that 11-digit IDs not born in 2000 are not modified."""
        original_id = "01026128186"
        df = pd.DataFrame({
            "IdNo": [original_id],  # 11 digits
            "Dob": [pd.to_datetime("02/01/1999")]  # Born 1999, not 2000
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should remain unchanged (not born in 2000)
        assert result_df.loc[0, "IdNo"] == original_id

    def test_no_repair_12_digit_id_not_born_2001_2009(self):
        """Test that 12-digit IDs not born 2001-2009 are not modified."""
        original_id = "1026128186"
        df = pd.DataFrame({
            "IdNo": [original_id],  # 12 digits
            "Dob": ["02/01/2010"]  # Born 2010, not 2001-2009
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should remain unchanged (not born 2001-2009)
        assert result_df.loc[0, "IdNo"] == original_id

    def test_invalid_dob_date(self):
        """Test handling of invalid DOB dates."""
        original_id = "01026128186"
        df = pd.DataFrame({
            "IdNo": [original_id],
            "Dob": [pd.NaT]  # Invalid date
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should remain unchanged (invalid DOB)
        assert result_df.loc[0, "IdNo"] == original_id

    def test_yymmdd_mismatch_no_repair(self):
        """Test that IDs with YYMMDD mismatch are not repaired."""
        # ID has YYMMDD "010261" but DOB is 02/01/2001 (YYMMDD "010201")
        original_id = "01026128186"
        df = pd.DataFrame({
            "IdNo": [original_id],  # 11 digits
            "Dob": ["01/02/2001"]  # Different YYMMDD
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should remain unchanged (YYMMDD mismatch)
        assert result_df.loc[0, "IdNo"] == original_id

    def test_mixed_scenarios(self):
        """Test multiple records with different scenarios."""
        df = pd.DataFrame({
            "IdNo": [
                "01026128186",  # 11 digits, born 2000 - should be repaired
                "101026128186",  # 12 digits, born 2001 - should be repaired
                "7612026128186",  # 13 digits - no repair needed
                "01026128186",  # 11 digits, born 1999 - no repair
                "101026128186"   # 12 digits, born 2010 - no repair
            ],
            "Dob": [
                "02/01/2000",  # 2000 - repair 11-digit
                "02/01/2001",  # 2001 - repair 12-digit
                "02/12/1976",  # 1976 - no repair needed
                "02/01/1999",  # 1999 - no repair for 11-digit
                "02/01/2010"   # 2010 - no repair for 12-digit
            ]
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Check each scenario
        assert result_df.loc[0, "IdNo"] == "0001026128186"  # Repaired: 11-digit + 2000
        assert result_df.loc[1, "IdNo"] == "0101026128186"   # Repaired: 12-digit + 2001
        assert result_df.loc[2, "IdNo"] == "7612026128186"  # Unchanged: 13-digit
        assert result_df.loc[3, "IdNo"] == "01026128186"   # Unchanged: 11-digit + 1999
        assert result_df.loc[4, "IdNo"] == "101026128186"    # Unchanged: 12-digit + 2010


    def test_none_values(self):
        """Test handling of None values."""
        df = pd.DataFrame({
            "IdNo": [None, "01026128186", ""],
            "Dob": ["02/01/2000", None, "02/01/2000"]
        })
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Only valid record should be processed
        assert pd.isna(result_df.loc[0, "IdNo"])  # None ID remains None
        assert result_df.loc[1, "IdNo"] == "01026128186"  # None DOB remains unchanged
        assert result_df.loc[2, "IdNo"] == ""  # Empty ID remains empty

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["IdNo", "Dob"])
        
        result_df = repair_sa_id_using_dob(df, "IdNo", "Dob")
        
        # Should return empty DataFrame unchanged
        assert result_df.empty
        assert list(result_df.columns) == ["IdNo", "Dob"]

    def test_missing_columns(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({"OtherColumn": [1, 2, 3]})
        
        with pytest.raises(KeyError):
            repair_sa_id_using_dob(df, "IdNo", "Dob")
