"""End-to-end tests: extract → verify DataFrame content."""

import tempfile
from pathlib import Path

from core import extract, write_excel

FIXTURES = Path(__file__).parent / "fixtures"


class TestEndToEnd:
    def test_clean_extraction(self):
        result = extract(FIXTURES / "header-at-1-without-summary.xlsx")
        df = result.dataframe
        assert len(df) == 20
        assert len(df.columns) == 56
        assert df.iloc[0]["EMPLOYEE NO"] == 20201001

    def test_extraction_with_summary(self):
        result = extract(FIXTURES / "header-at-5-with-summary.xlsx")
        df = result.dataframe
        assert len(df) == 255
        assert len(df.columns) == 20
        assert df.iloc[0]["First names"] == "NAME 1"


