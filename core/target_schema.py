"""Target schema definition -- single source of truth for column names and rules.

To add a new column: add a Column entry to TABLE_SPEC.
To add a cleaning rule: define a function and attach it as a Rule to the column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional
from core.constant import Gender

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    id: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]


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
                df = rule.apply(df)
                applied.append(rule.id)
        return df, applied


# ── Rule functions ────────────────────────────────────────────────────────

def validate_email(df: pd.DataFrame, check_deliverability=False) -> pd.DataFrame:
    """Validate email addresses using email-validator library."""
    from email_validator import validate_email, EmailNotValidError

    if "PersonalEmailAddress" in df.columns:
        mask = df["PersonalEmailAddress"].notna()
        invalid_count = 0

        for idx in df[mask].index:
            try:
                validate_email(df.loc[idx, "PersonalEmailAddress"], check_deliverability=check_deliverability)
            except EmailNotValidError:
                df.loc[idx, "PersonalEmailAddress"] = None
                invalid_count += 1

        logger.info(f"Validated emails: {invalid_count} invalid emails cleared")

    return df


def _derive_monthly_from_annual(df: pd.DataFrame) -> pd.DataFrame:
    monthly_null = df["MonthlyIncome"].isna()
    annual_avail = df["AnnualIncome"].notna()

    # Only derive if we have valid annual values
    if annual_avail.any():
        annual_values = df.loc[annual_avail, "AnnualIncome"]
        # Convert to numeric, handling None values, and round to 2 decimal places
        annual_numeric = pd.to_numeric(annual_values, errors='coerce').round(2)
        # Only update where conversion succeeded
        valid_mask = annual_numeric.notna()
        if valid_mask.any():
            df.loc[monthly_null & annual_avail, "MonthlyIncome"] = (annual_numeric[valid_mask] / 12).round(2)
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
        # Convert to numeric, handling None values, and round to 2 decimal places
        monthly_numeric = pd.to_numeric(monthly_values, errors='coerce').round(2)
        # Only update where conversion succeeded
        valid_mask = monthly_numeric.notna()
        if valid_mask.any():
            df.loc[annual_null & monthly_avail, "AnnualIncome"] = (monthly_numeric[valid_mask] * 12).round(2)
    else:
        # If no monthly values at all, ensure AnnualIncome stays as is (could be empty or have values)
        pass

    return df


_GENDER_MAP = {
    "f": Gender.FEMALE.value, "female": Gender.FEMALE.value,
    "m": Gender.MALE.value, "male": Gender.MALE.value,
}


def _normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    original = df["Gender"]
    normalized = original.astype(str).str.strip().str.lower().map(_GENDER_MAP)
    df["Gender"] = normalized.where(normalized.notna(), original)
    return df


def _sanitize_mobile_number(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize mobile phone numbers by padding missing leading zeros."""
    if "CellPhoneNumber" in df.columns:
        mask = df["CellPhoneNumber"].notna()

        for idx in df[mask].index:
            phone = str(df.loc[idx, "CellPhoneNumber"]).strip()

            # Remove any non-digit characters
            digits_only = ''.join(filter(str.isdigit,
                                         phone))

            # Pad 9-digit numbers that don't start with 0
            if len(digits_only) == 9 and not digits_only.startswith('0'):
                digits_only = '0' + digits_only

            df.loc[idx, "CellPhoneNumber"] = digits_only

        logger.info(f"Sanitized mobile numbers for {mask.sum()} records")

    return df


def repair_sa_id_candidate(ids, dob):
    """
    Generate repaired SA IDs based on DOB rules.
    Does not validate DOB correctness.
    """

    ids = ids.astype(str)
    id_len = ids.str.len()

    dob_year = dob.dt.year
    expected_yymmdd = dob.dt.strftime("%y%m%d")
    expected_ymmdd = expected_yymmdd.str[1:]
    expected_mmdd = expected_yymmdd.str[2:]

    repaired_ids = ids.copy()

    mask_11 = (
        (id_len == 11)
        & (dob_year == 2000)
        & (ids.str[:4] == expected_mmdd)
    )

    repaired_ids.loc[mask_11] = "00" + ids.loc[mask_11]

    mask_12 = (
        (~mask_11)
        & (id_len == 12)
        & (dob_year.between(2001, 2009))
        & (ids.str[:5] == expected_ymmdd)
    )

    repaired_ids.loc[mask_12] = "0" + ids.loc[mask_12]

    return repaired_ids


def validate_id_matches_dob(ids, dob):
    """
    Validate YYMMDD in SA ID against DOB column.
    """

    expected_yymmdd = dob.dt.strftime("%y%m%d")
    yymmdd_from_id = ids.str[:6]

    dob_match = yymmdd_from_id == expected_yymmdd

    extracted_date = pd.to_datetime(
        yymmdd_from_id,
        format="%y%m%d",
        errors="coerce"
    )

    valid_date = extracted_date.notna()

    return dob_match & valid_date

def repair_sa_id_using_dob(df, id_col, dob_col):

    df = df.copy()

    ids = df[id_col].astype(str)

    dob = pd.to_datetime(
        df[dob_col],
        errors="coerce",
        dayfirst=True
    )

    repaired_ids = repair_sa_id_candidate(ids, dob)

    valid_mask = validate_id_matches_dob(repaired_ids, dob)

    df.loc[valid_mask, id_col] = repaired_ids[valid_mask]

    return df

def _derive_category(df):
    """Extract numeric category from various category string formats."""
    if "Category" not in df.columns:
        return df
    
    mask = df["Category"].notna()
    
    for idx in df[mask].index:
        category_str = str(df.loc[idx, "Category"]).strip()
        
        # Extract numbers from the string
        import re
        numbers = re.findall(r'\d+', category_str)
        
        if numbers:
            # Take the first number found
            df.loc[idx, "Category"] = int(numbers[0])
        else:
            # If no numbers found, set to None
            df.loc[idx, "Category"] = None
    
    return df


# ── Table specification ───────────────────────────────────────────────────
# Add all target columns here.  Columns without rules use an empty list.
# Todo: Validate the output data spec

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
    rules=[Rule("sanitize_id_number", partial(repair_sa_id_using_dob, id_col="IdNo", dob_col="Dob"))],
    aliases=["id number", "idno", "member id", "identification number", "id", "identity number"],
    description="Unique identifier for the person (13-digit South African ID or some similar id)",
    data_type="object",
    excel_format="@",
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
    rules=[Rule("sanitize_mobile_number", _sanitize_mobile_number)],
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
    data_type="str",
    rules=[Rule("validate_email", validate_email)]
)
Category = Column(name="Category", data_type="int64", rules=[Rule("derive_category", _derive_category)])
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
        Flex,
        Flextype,
        PHIWaiver,
        MIIBWaiver,
        TTDWaiver,
        Remark,
        GLA,
        GLAFlex,
        LumpSumDisability,
        DisabilityIncomeBenefit,
        TemporaryDisability,
        SGLA,
        SLSDB,
        TraumaStandard,
        TraumaComprehensive,
        AccidentDeath,
        AccidentBLoss,
        AccidentDBLoss,
        FuneralCover,
        EmployerWaiver,
        SalaryTopUp,
        GlobalEducator,
        MAPW,

    ]
)
