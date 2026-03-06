"""Target schema definition -- single source of truth for column names and rules.

To add a new column: add a Column entry to TABLE_SPEC.
To add a cleaning rule: define a function and attach it as a Rule to the column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    id: str
    fn: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class Column:
    name: str
    rules: List[Rule] = field(default_factory=list)


@dataclass
class TableSpec:
    columns: List[Column]

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Run all applicable column rules. Returns (cleaned_df, applied_rule_ids)."""
        applied: List[str] = []
        for col in self.columns:
            if col.name not in df.columns:
                continue
            for rule in col.rules:
                logger.debug(f"Running rule '{rule.id}' on column '{col.name}'")
                df = rule.fn(df)
                applied.append(rule.id)
        return df, applied


# ── Rule functions ────────────────────────────────────────────────────────

def _derive_monthly_from_annual(df: pd.DataFrame) -> pd.DataFrame:
    monthly_null = df["MonthlyIncome"].isna()
    annual_avail = df["AnnualIncome"].notna()
    
    # Only derive if we have valid annual values
    if annual_avail.any():
        annual_values = df.loc[annual_avail, "AnnualIncome"]
        # Convert to numeric, handling None values
        annual_numeric = pd.to_numeric(annual_values, errors='coerce')
        # Only update where conversion succeeded
        valid_mask = annual_numeric.notna()
        if valid_mask.any():
            df.loc[monthly_null & annual_avail, "MonthlyIncome"] = annual_numeric[valid_mask] / 12
    else:
        # If no annual values at all, ensure MonthlyIncome stays as is (could be empty or have values)
        pass
    
    return df


def _derive_annual_from_monthly(df: pd.DataFrame) -> pd.DataFrame:
    annual_null = df["AnnualIncome"].isna()
    monthly_avail = df["MonthlyIncome"].notna()
    
    # Only derive if we have valid monthly values
    if monthly_avail.any():
        monthly_values = df.loc[monthly_avail, "MonthlyIncome"]
        # Convert to numeric, handling None values
        monthly_numeric = pd.to_numeric(monthly_values, errors='coerce')
        # Only update where conversion succeeded
        valid_mask = monthly_numeric.notna()
        if valid_mask.any():
            df.loc[annual_null & monthly_avail, "AnnualIncome"] = monthly_numeric[valid_mask] * 12
    else:
        # If no monthly values at all, ensure AnnualIncome stays as is (could be empty or have values)
        pass
    
    return df


_GENDER_MAP = {
    "f": "Female", "female": "Female",
    "m": "Male", "male": "Male",
}


def _normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    original = df["Gender"]
    normalized = original.astype(str).str.strip().str.lower().map(_GENDER_MAP)
    df["Gender"] = normalized.where(normalized.notna(), original)
    return df


# ── Table specification ───────────────────────────────────────────────────
# Add all target columns here.  Columns without rules use an empty list.

FirstName = Column(name="FirstName", rules=[])
Surname = Column(name="Surname", rules=[])
Gender = Column(name="Gender", rules=[Rule("normalize_gender", _normalize_gender)])
Dob = Column(name="Dob", rules=[])
IdNo = Column(name="IdNo", rules=[])
AnnualIncome = Column(name="AnnualIncome", rules=[Rule("derive_annual_from_monthly", _derive_annual_from_monthly)])
MonthlyIncome = Column(name="MonthlyIncome", rules=[Rule("derive_monthly_from_annual", _derive_monthly_from_annual)])
Category = Column(name="Category", rules=[])
PassportCountryofIssue = Column(name="PassportCountryofIssue", rules=[])
PaypointorBranchName = Column(name="PaypointorBranchName", rules=[])
PayrollNumber = Column(name="PayrollNumber", rules=[])
CellPhoneNumber = Column(name="CellPhoneNumber", rules=[])
PersonalEmailAddress = Column(name="PersonalEmailAddress", rules=[])
ResidentialAddressLine1 = Column(name="ResidentialAddressLine1", rules=[])
ResidentialAddressLine2 = Column(name="ResidentialAddressLine2", rules=[])
ResidentialAddressLine3 = Column(name="ResidentialAddressLine3", rules=[])
ResidentialPostCode = Column(name="ResidentialPostCode", rules=[])
ResidentialCity = Column(name="ResidentialCity", rules=[])
ResidentialProvince = Column(name="ResidentialProvince", rules=[])
ResidentialCountry = Column(name="ResidentialCountry", rules=[])
JobDescription = Column(name="JobDescription", rules=[])
MaritalStatus = Column(name="MaritalStatus", rules=[])
SpouseFirstName = Column(name="SpouseFirstName", rules=[])
SpouseLastName = Column(name="SpouseLastName", rules=[])
SpouseGender = Column(name="SpouseGender", rules=[])
SpouseDateofBirth = Column(name="SpouseDateofBirth", rules=[])
SpouseIDorPassportNumber = Column(name="SpouseIDorPassportNumber", rules=[])
Membergroup = Column(name="Membergroup", rules=[])
Flex = Column(name="Flex", rules=[])
Flextype = Column(name="Flextype", rules=[])
PHIWaiver = Column(name="PHIWaiver", rules=[])
MIIBWaiver = Column(name="MIIBWaiver", rules=[])
TTDWaiver = Column(name="TTDWaiver", rules=[])
Remark = Column(name="Remark", rules=[])
GLA = Column(name="GLA", rules=[])
GLAFlex = Column(name="GLAFlex", rules=[])
LumpSumDisability = Column(name="LumpSumDisability", rules=[])
DisabilityIncomeBenefit = Column(name="DisabilityIncomeBenefit", rules=[])
TemporaryDisability = Column(name="TemporaryDisability", rules=[])
SGLA = Column(name="SGLA", rules=[])
SLSDB = Column(name="SLSDB", rules=[])
TraumaStandard = Column(name="TraumaStandard", rules=[])
TraumaComprehensive = Column(name="TraumaComprehensive", rules=[])
AccidentDeath = Column(name="AccidentDeath", rules=[])
AccidentBLoss = Column(name="AccidentBLoss", rules=[])
AccidentDBLoss = Column(name="AccidentDBLoss", rules=[])
FuneralCover = Column(name="FuneralCover", rules=[])
EmployerWaiver = Column(name="EmployerWaiver", rules=[])
SalaryTopUp = Column(name="SalaryTopUp", rules=[])
GlobalEducator = Column(name="GlobalEducator", rules=[])
MAPW = Column(name="MAPW", rules=[])

TABLE_SPEC = TableSpec(
    [
        FirstName,
        Surname,
        Gender,
        Dob,
        IdNo,
        AnnualIncome,
        MonthlyIncome,
        Category,
        PassportCountryofIssue,
        PaypointorBranchName,
        PayrollNumber,
        CellPhoneNumber,
        PersonalEmailAddress,
        # ResidentialAddressLine1,
        # ResidentialAddressLine2,
        # ResidentialAddressLine3,
        # ResidentialPostCode,
        # ResidentialCity,
        # ResidentialProvince,
        # ResidentialCountry,
        # JobDescription,
        # MaritalStatus,
        # SpouseFirstName,
        # SpouseLastName,
        # SpouseGender,
        # SpouseDateofBirth,
        # SpouseIDorPassportNumber,
        # Membergroup,
        # Flex,
        # Flextype,
        # PHIWaiver,
        # MIIBWaiver,
        # TTDWaiver,
        # Remark,
        # GLA,
        # GLAFlex,
        # LumpSumDisability,
        # DisabilityIncomeBenefit,
        # TemporaryDisability,
        # SGLA,
        # SLSDB,
        # TraumaStandard,
        # TraumaComprehensive,
        # AccidentDeath,
        # AccidentBLoss,
        # AccidentDBLoss,
        # FuneralCover,
        # EmployerWaiver,
        # SalaryTopUp,
        # GlobalEducator,
        # MAPW,

    ]
)
