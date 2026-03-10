"""Target schema definition -- single source of truth for column names and rules.

To add a new column: add a Column entry to TABLE_SPEC.
To add a cleaning rule: define a function and attach it as a Rule to the column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

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
    aliases: List[str] = field(default_factory=list)  # Alternative column names (optional)
    description: str = ""  # What this column represents (optional)
    data_type: Optional[str] = None  # Expected pandas dtype (e.g., 'object', 'int64', 'float64', 'datetime64[ns]')
    excel_format: Optional[str] = None  # Excel format (e.g., 'R #,##0.00', 'dd/mm/yyyy', '#,##0')
    ignore_number_stored_as_text: bool = False

    def get_all_names(self) -> List[str]:
        """Return all searchable names for this column (name + aliases)."""
        return [self.name] + self.aliases

    def get_context_string(self) -> str:
        """Return formatted context string for LLM prompts."""
        parts = []
        if self.aliases:
            parts.append(f"Aliases: {', '.join(self.aliases)}")
        if self.description:
            parts.append(f"Description: {self.description}")
        # if self.data_type:
        #     parts.append(f"Data Type: {self.data_type}")
        # if self.excel_format:
        #     parts.append(f"Excel Format: {self.excel_format}")
        return " | ".join(parts) if parts else f"Column: {self.name}"


@dataclass
class TableSpec:
    columns: List[Column]

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get_llm_context(self) -> str:
        """Return formatted context string for LLM prompts with all columns."""
        context_lines = ["Target Column Mapping Context:"]
        for col in self.columns:
            context_lines.append(f"- {col.name}: {col.get_context_string()}")
        return "\n".join(context_lines)

    def get_column_by_name(self, name: str) -> Optional[Column]:
        """Find column by name or alias."""
        for col in self.columns:
            if name.lower() in [n.lower() for n in col.get_all_names()]:
                return col
        return None

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

FirstName = Column(
    name="FirstName",
    aliases=["first name", "given name"],
    description="Person's first or given name. In case source does not exist,Should be split if full name is given",
    data_type="object"
)

Surname = Column(
    name="Surname",
    aliases=["last name", "family name", "surname"],
    description="Person's last or family name.  In case source does not exist,Should be split if full name is given",
    data_type="object"
)

Gender = Column(
    name="Gender",
    rules=[Rule("normalize_gender", _normalize_gender)],
    aliases=["sex", "gender"],
    description="Person's gender (Male/Female)",
    data_type="object"
)

Dob = Column(
    name="Dob",
    rules=[],
    aliases=["date of birth", "dob", "birth date", "birthdate"],
    description="Person's date of birth",
    data_type="datetime64[ns]",
    excel_format="dd/mm/yyyy"
    # excel_format="[$-en-ZA]dd/mm/yyyy"
)

IdNo = Column(
    name="IdNo",
    rules=[],
    aliases=["id number", "idno", "member id", "identification number", "id", "identity number"],
    description="Unique identifier for the person (13-digit South African ID or some similar id)",
    data_type="object"
)
AnnualIncome = Column(
    name="AnnualIncome",
    rules=[Rule("derive_annual_from_monthly", _derive_annual_from_monthly)],
    aliases=["annual income", "yearly salary", "annual salary", "yearly income"],
    description="Person's annual income in South African Rand",
    data_type="float64",
    # excel_format='_([$$-409]* #,##0.00_)' # changes accordingly based on the excel locale
    excel_format='_([$R-en-ZA]* #,##0.00_)'
)

MonthlyIncome = Column(
    name="MonthlyIncome",
    rules=[Rule("derive_monthly_from_annual", _derive_monthly_from_annual)],
    aliases=["monthly income", "monthly salary", "monthly pay"],
    description="Person's monthly income in South African Rand",
    data_type="float64",
    excel_format='_([$R-en-ZA]* #,##0.00_)'
)

CellPhoneNumber = Column(
    name="CellPhoneNumber",
    aliases=["cell phone", "mobile number", "cellphone", "mobile", "phone number"],
    description="Person's mobile/cell phone number",
    data_type="str",
    excel_format="@",
    ignore_number_stored_as_text=True
)

PersonalEmailAddress = Column(
    name="PersonalEmailAddress",
    aliases=["email", "email address", "personal email", "mail"],
    description="Person's personal email address",
    data_type="str"
)
Category = Column(name="Category")
PassportCountryofIssue = Column(name="PassportCountryofIssue")
PaypointorBranchName = Column(name="PaypointorBranchName")
PayrollNumber = Column(name="PayrollNumber")
ResidentialAddressLine1 = Column(name="ResidentialAddressLine1")
ResidentialAddressLine2 = Column(name="ResidentialAddressLine2")
ResidentialAddressLine3 = Column(name="ResidentialAddressLine3")
ResidentialPostCode = Column(name="ResidentialPostCode")
ResidentialCity = Column(name="ResidentialCity")
ResidentialProvince = Column(name="ResidentialProvince")
ResidentialCountry = Column(name="ResidentialCountry")
JobDescription = Column(name="JobDescription")
MaritalStatus = Column(name="MaritalStatus")
SpouseFirstName = Column(name="SpouseFirstName")
SpouseLastName = Column(name="SpouseLastName")
SpouseGender = Column(name="SpouseGender")
SpouseDateofBirth = Column(name="SpouseDateofBirth")
SpouseIDorPassportNumber = Column(name="SpouseIDorPassportNumber")
Membergroup = Column(name="Membergroup")
Flex = Column(name="Flex")
Flextype = Column(name="Flextype")
PHIWaiver = Column(name="PHIWaiver")
MIIBWaiver = Column(name="MIIBWaiver")
TTDWaiver = Column(name="TTDWaiver")
Remark = Column(name="Remark")
GLA = Column(name="GLA")
GLAFlex = Column(name="GLAFlex")
LumpSumDisability = Column(name="LumpSumDisability")
DisabilityIncomeBenefit = Column(name="DisabilityIncomeBenefit")
TemporaryDisability = Column(name="TemporaryDisability")
SGLA = Column(name="SGLA")
SLSDB = Column(name="SLSDB")
TraumaStandard = Column(name="TraumaStandard")
TraumaComprehensive = Column(name="TraumaComprehensive")
AccidentDeath = Column(name="AccidentDeath")
AccidentBLoss = Column(name="AccidentBLoss")
AccidentDBLoss = Column(name="AccidentDBLoss")
FuneralCover = Column(name="FuneralCover")
EmployerWaiver = Column(name="EmployerWaiver")
SalaryTopUp = Column(name="SalaryTopUp")
GlobalEducator = Column(name="GlobalEducator")
MAPW = Column(name="MAPW")

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
        ResidentialAddressLine1,
        ResidentialAddressLine2,
        ResidentialAddressLine3,
        ResidentialPostCode,
        ResidentialCity,
        ResidentialProvince,
        ResidentialCountry,
        JobDescription,
        MaritalStatus,
        SpouseFirstName,
        SpouseLastName,
        SpouseGender,
        SpouseDateofBirth,
        SpouseIDorPassportNumber,
        Membergroup,
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
