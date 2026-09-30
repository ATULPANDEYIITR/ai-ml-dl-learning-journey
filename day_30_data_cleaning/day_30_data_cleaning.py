"""
Data Cleaning Laboratory
========================

A self-contained Python implementation covering four distinct data-cleaning
problems:

- Missing values
- Duplicate records
- Outliers
- Inconsistent values

The implementation uses only Python's standard library and demonstrates
validation, profiling, normalization, deterministic cleaning rules,
outlier detection, audit trails, quality metrics, and CSV persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable
import csv
import math
import re


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CleaningIssue:
    """Records one transformation or data-quality finding."""

    record_id: str
    field: str
    issue_type: str
    original_value: Any
    cleaned_value: Any
    action: str
    reason: str


@dataclass
class CleaningReport:
    """Aggregates profiling and transformation results."""

    issues: list[CleaningIssue] = field(default_factory=list)

    def add(
        self,
        record_id: str,
        field: str,
        issue_type: str,
        original_value: Any,
        cleaned_value: Any,
        action: str,
        reason: str,
    ) -> None:
        self.issues.append(
            CleaningIssue(
                record_id=record_id,
                field=field,
                issue_type=issue_type,
                original_value=original_value,
                cleaned_value=cleaned_value,
                action=action,
                reason=reason,
            )
        )

    def count(self, issue_type: str) -> int:
        return sum(issue.issue_type == issue_type for issue in self.issues)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Sample dataset
# ---------------------------------------------------------------------------

def build_sample_data() -> list[dict[str, Any]]:
    """
    Create a realistic customer transaction dataset containing deliberate
    quality problems.

    The duplicate transaction demonstrates record-level duplication.
    Missing values include blank strings and None.
    Outliers are represented by an unusually large transaction amount.
    Inconsistent values include country, state, category, date, and email
    formatting variations.
    """

    return [
        {
            "transaction_id": "TX1001",
            "customer_name": "  Priya Sharma ",
            "email": "PRIYA.SHARMA@EXAMPLE.COM ",
            "country": "india",
            "state": "uttar pradesh",
            "category": " electronics ",
            "amount": "12500.00",
            "transaction_date": "2026-09-01",
        },
        {
            "transaction_id": "TX1002",
            "customer_name": "Rahul Verma",
            "email": "rahul.verma@example.com",
            "country": "IND",
            "state": "UP",
            "category": "Electronics",
            "amount": "13,250",
            "transaction_date": "01/09/2026",
        },
        {
            "transaction_id": "TX1003",
            "customer_name": "Neha Singh",
            "email": "neha.singh@example.com",
            "country": "India ",
            "state": "U.P.",
            "category": "Home & Kitchen",
            "amount": "",
            "transaction_date": "2026-09-03",
        },
        {
            "transaction_id": "TX1004",
            "customer_name": "Amit Kumar",
            "email": "amit.kumar@example.com",
            "country": "IN",
            "state": "Delhi",
            "category": "home and kitchen",
            "amount": "8900",
            "transaction_date": None,
        },
        {
            "transaction_id": "TX1005",
            "customer_name": "Sara Khan",
            "email": "sara.khan@example.com",
            "country": "India",
            "state": "delhi",
            "category": "HOME & KITCHEN",
            "amount": "9100",
            "transaction_date": "2026-09-05",
        },
        {
            "transaction_id": "TX1006",
            "customer_name": "Vikram Rao",
            "email": "vikram.rao@example.com",
            "country": "United States",
            "state": "California",
            "category": "Electronics",
            "amount": "1500000",
            "transaction_date": "2026-09-06",
        },
        {
            "transaction_id": "TX1007",
            "customer_name": "Meera Joshi",
            "email": "MEERA.JOSHI@example.com",
            "country": "USA",
            "state": "CA",
            "category": "Electronics",
            "amount": "11750",
            "transaction_date": "2026-09-07",
        },
        {
            "transaction_id": "TX1008",
            "customer_name": "Arjun Patel",
            "email": "arjun.patel@example.com",
            "country": "INDIA",
            "state": "Gujarat",
            "category": "Electronics",
            "amount": "12100",
            "transaction_date": "07-09-2026",
        },
        {
            "transaction_id": "TX1009",
            "customer_name": "Kabir Mehta",
            "email": "kabir.mehta@example.com",
            "country": "IN",
            "state": "MH",
            "category": "Grocery",
            "amount": "2150",
            "transaction_date": "2026-09-08",
        },
        {
            "transaction_id": "TX1009",
            "customer_name": "Kabir Mehta",
            "email": "kabir.mehta@example.com",
            "country": "IN",
            "state": "MH",
            "category": "Grocery",
            "amount": "2150",
            "transaction_date": "2026-09-08",
        },
        {
            "transaction_id": "TX1010",
            "customer_name": "Ananya Das",
            "email": "ananya.das@example.com",
            "country": "India",
            "state": "West Bengal",
            "category": "groceries",
            "amount": "1980",
            "transaction_date": "2026-09-09",
        },
        {
            "transaction_id": "TX1011",
            "customer_name": "Rohan Gupta",
            "email": "rohan.gupta@example.com",
            "country": "IND",
            "state": "Karnataka",
            "category": "Grocery",
            "amount": "2250",
            "transaction_date": "2026-09-10",
        },
        {
            "transaction_id": "TX1012",
            "customer_name": "Isha Roy",
            "email": "isha.roy@example.com",
            "country": "India",
            "state": "WB",
            "category": "Groceries",
            "amount": "2100",
            "transaction_date": "2026-09-11",
        },
    ]


# ---------------------------------------------------------------------------
# General profiling
# ---------------------------------------------------------------------------

def is_missing(value: Any) -> bool:
    """Treat None and whitespace-only strings as missing."""

    if value is None:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    return False


def profile_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate field-level completeness and cardinality statistics."""

    fields: set[str] = set()
    for row in rows:
        fields.update(row.keys())

    profile: dict[str, Any] = {
        "row_count": len(rows),
        "field_count": len(fields),
        "fields": {},
    }

    for field_name in sorted(fields):
        values = [row.get(field_name) for row in rows]
        missing = sum(is_missing(value) for value in values)
        unique = len(
            {
                str(value).strip().lower()
                for value in values
                if not is_missing(value)
            }
        )

        profile["fields"][field_name] = {
            "missing": missing,
            "missing_rate": missing / len(rows) if rows else 0.0,
            "unique_non_missing": unique,
        }

    return profile


def print_profile(profile: dict[str, Any]) -> None:
    print("\nDATASET PROFILE")
    print("=" * 72)
    print(f"Rows:   {profile['row_count']}")
    print(f"Fields: {profile['field_count']}")
    print("\nField quality:")
    for name, stats in profile["fields"].items():
        print(
            f"  {name:18} "
            f"missing={stats['missing']:2} "
            f"missing_rate={stats['missing_rate']:.1%} "
            f"unique={stats['unique_non_missing']:2}"
        )


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def parse_amount(value: Any) -> float | None:
    """Convert common currency-like representations into a numeric value."""

    if is_missing(value):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        text = text.replace(",", "")
        text = re.sub(r"[₹$€£]", "", text)

        try:
            number = float(text)
        except ValueError:
            return None

    if not math.isfinite(number):
        return None

    return number


def impute_missing_amounts(
    rows: list[dict[str, Any]],
    report: CleaningReport,
) -> None:
    """
    Fill missing transaction amounts with the median of valid amounts.

    Median is used because transaction amounts can be skewed and therefore
    a mean can be disproportionately influenced by unusually large values.
    """

    values = [
        parsed
        for row in rows
        if (parsed := parse_amount(row.get("amount"))) is not None
    ]

    if not values:
        raise ValueError("No valid amount values are available for imputation.")

    replacement = median(values)

    for row in rows:
        original = row.get("amount")

        if is_missing(original):
            row["amount"] = round(replacement, 2)
            report.add(
                row["transaction_id"],
                "amount",
                "missing_value",
                original,
                row["amount"],
                "median_imputation",
                "Missing transaction amount replaced with the dataset median.",
            )


def handle_missing_dates(
    rows: list[dict[str, Any]],
    report: CleaningReport,
) -> None:
    """
    Missing dates are not imputed because a fabricated transaction date could
    alter reporting periods. They are explicitly marked for downstream review.
    """

    for row in rows:
        if is_missing(row.get("transaction_date")):
            original = row.get("transaction_date")
            row["transaction_date"] = None
            report.add(
                row["transaction_id"],
                "transaction_date",
                "missing_value",
                original,
                None,
                "retain_missing",
                "Date is semantically important and cannot be safely inferred.",
            )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def normalize_for_duplicate_detection(value: Any) -> str:
    """Create a conservative comparison representation."""

    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value).strip().casefold())


def duplicate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """
    Build a business-oriented duplicate key.

    Transaction ID alone is sufficient for this dataset because it is treated
    as a unique business identifier. Email is retained as a secondary signal.
    """

    return (
        normalize_for_duplicate_detection(row.get("transaction_id")),
        normalize_for_duplicate_detection(row.get("email")),
        normalize_for_duplicate_detection(row.get("transaction_date")),
    )


def remove_exact_business_duplicates(
    rows: list[dict[str, Any]],
    report: CleaningReport,
) -> list[dict[str, Any]]:
    """Keep the first occurrence of each business-level duplicate."""

    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        key = duplicate_key(row)

        if key in seen:
            report.add(
                row["transaction_id"],
                "__record__",
                "duplicate",
                dict(row),
                None,
                "remove_duplicate",
                "A previously retained record has the same business identity.",
            )
            continue

        seen.add(key)
        cleaned.append(row)

    return cleaned


# ---------------------------------------------------------------------------
# Inconsistent values
# ---------------------------------------------------------------------------

COUNTRY_MAP = {
    "india": "India",
    "ind": "India",
    "in": "India",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
}

STATE_MAP = {
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "uttar pradesh": "Uttar Pradesh",
    "delhi": "Delhi",
    "ca": "California",
    "california": "California",
    "gujarat": "Gujarat",
    "mh": "Maharashtra",
    "maharashtra": "Maharashtra",
    "wb": "West Bengal",
    "west bengal": "West Bengal",
    "karnataka": "Karnataka",
}

CATEGORY_MAP = {
    "electronics": "Electronics",
    "home & kitchen": "Home & Kitchen",
    "home and kitchen": "Home & Kitchen",
    "grocery": "Grocery",
    "groceries": "Grocery",
}


def normalize_text(value: Any) -> str | None:
    """Normalize surrounding whitespace without inventing a semantic value."""

    if is_missing(value):
        return None

    return re.sub(r"\s+", " ", str(value).strip())


def normalize_email(value: Any) -> str | None:
    """Normalize email casing and whitespace."""

    normalized = normalize_text(value)

    if normalized is None:
        return None

    return normalized.casefold()


def normalize_lookup(
    value: Any,
    mapping: dict[str, str],
) -> str | None:
    """Map known aliases to a canonical value."""

    normalized = normalize_text(value)

    if normalized is None:
        return None

    return mapping.get(normalized.casefold(), normalized)


def normalize_transaction_date(value: Any) -> str | None:
    """
    Convert supported date formats to ISO YYYY-MM-DD.

    Ambiguous dates are not silently guessed. Only formats with an explicit
    four-digit year and unambiguous day/month positions are accepted here.
    """

    if is_missing(value):
        return None

    text = str(value).strip()

    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    )

    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unsupported transaction date format: {value!r}")


def standardize_inconsistent_values(
    rows: list[dict[str, Any]],
    report: CleaningReport,
) -> None:
    """Apply explicit domain mappings to categorical and formatted fields."""

    for row in rows:
        record_id = row["transaction_id"]

        transformations = {
            "customer_name": normalize_text(row.get("customer_name")),
            "email": normalize_email(row.get("email")),
            "country": normalize_lookup(row.get("country"), COUNTRY_MAP),
            "state": normalize_lookup(row.get("state"), STATE_MAP),
            "category": normalize_lookup(row.get("category"), CATEGORY_MAP),
        }

        for field_name, cleaned in transformations.items():
            original = row.get(field_name)

            if original != cleaned:
                row[field_name] = cleaned
                report.add(
                    record_id,
                    field_name,
                    "inconsistent_value",
                    original,
                    cleaned,
                    "canonicalize",
                    "Known formatting or domain alias converted to a canonical representation.",
                )

        original_date = row.get("transaction_date")

        try:
            cleaned_date = normalize_transaction_date(original_date)
        except ValueError as exc:
            report.add(
                record_id,
                "transaction_date",
                "inconsistent_value",
                original_date,
                None,
                "reject_invalid_format",
                str(exc),
            )
            row["transaction_date"] = None
        else:
            if original_date != cleaned_date:
                row["transaction_date"] = cleaned_date
                report.add(
                    record_id,
                    "transaction_date",
                    "inconsistent_value",
                    original_date,
                    cleaned_date,
                    "canonicalize",
                    "Date converted to ISO 8601 calendar representation.",
                )

        parsed_amount = parse_amount(row.get("amount"))
        if parsed_amount is not None and row.get("amount") != parsed_amount:
            original_amount = row.get("amount")
            row["amount"] = parsed_amount
            report.add(
                record_id,
                "amount",
                "inconsistent_value",
                original_amount,
                parsed_amount,
                "normalize_numeric_format",
                "Currency separators and textual numeric formatting were normalized.",
            )


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def quartiles(values: list[float]) -> tuple[float, float]:
    """Calculate Q1 and Q3 using the median-of-halves method."""

    if len(values) < 4:
        raise ValueError("At least four values are required for IQR analysis.")

    ordered = sorted(values)
    midpoint = len(ordered) // 2

    if len(ordered) % 2 == 0:
        lower = ordered[:midpoint]
        upper = ordered[midpoint:]
    else:
        lower = ordered[:midpoint]
        upper = ordered[midpoint + 1 :]

    return median(lower), median(upper)


def detect_iqr_outliers(
    rows: list[dict[str, Any]],
    field_name: str,
    report: CleaningReport,
    multiplier: float = 1.5,
) -> list[dict[str, Any]]:
    """
    Detect statistical outliers using the interquartile range.

    Detection does not automatically delete or cap values. The issue is
    recorded while the original observation remains available for review.
    """

    numeric_rows: list[tuple[dict[str, Any], float]] = []

    for row in rows:
        value = parse_amount(row.get(field_name))
        if value is not None:
            numeric_rows.append((row, value))

    values = [value for _, value in numeric_rows]

    if len(values) < 4:
        return []

    q1, q3 = quartiles(values)
    iqr = q3 - q1

    if iqr == 0:
        return []

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers: list[dict[str, Any]] = []

    for row, value in numeric_rows:
        if value < lower_bound or value > upper_bound:
            outliers.append(row)
            report.add(
                row["transaction_id"],
                field_name,
                "outlier",
                value,
                value,
                "flag_only",
                (
                    f"IQR={iqr:.2f}, lower_bound={lower_bound:.2f}, "
                    f"upper_bound={upper_bound:.2f}; observation requires review."
                ),
            )

    return outliers


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
)


def validate_clean_dataset(rows: list[dict[str, Any]]) -> list[str]:
    """
    Apply post-cleaning data-quality constraints.

    The validator deliberately reports errors rather than silently changing
    data because validation and transformation are separate responsibilities.
    """

    errors: list[str] = []
    seen_ids: set[str] = set()

    for row in rows:
        record_id = row.get("transaction_id")

        if not record_id:
            errors.append("Record is missing transaction_id.")
        elif record_id in seen_ids:
            errors.append(f"Duplicate transaction_id remains: {record_id}")
        else:
            seen_ids.add(record_id)

        email = row.get("email")
        if email is not None and not EMAIL_PATTERN.match(email):
            errors.append(
                f"{record_id}: invalid email after normalization: {email!r}"
            )

        amount = row.get("amount")
        if not isinstance(amount, (int, float)) or not math.isfinite(amount):
            errors.append(f"{record_id}: amount is not a finite number.")
        elif amount < 0:
            errors.append(f"{record_id}: negative amount is not permitted.")

        country = row.get("country")
        if country not in {"India", "United States"}:
            errors.append(
                f"{record_id}: unsupported canonical country {country!r}"
            )

        category = row.get("category")
        if category not in {"Electronics", "Home & Kitchen", "Grocery"}:
            errors.append(
                f"{record_id}: unsupported canonical category {category!r}"
            )

        date_value = row.get("transaction_date")
        if date_value is not None:
            try:
                datetime.strptime(date_value, "%Y-%m-%d")
            except ValueError:
                errors.append(
                    f"{record_id}: transaction_date is not ISO formatted."
                )

    return errors


# ---------------------------------------------------------------------------
# Audit and persistence
# ---------------------------------------------------------------------------

def write_clean_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Persist the cleaned dataset with deterministic field ordering."""

    if not rows:
        raise ValueError("Cannot write an empty dataset.")

    fieldnames = [
        "transaction_id",
        "customer_name",
        "email",
        "country",
        "state",
        "category",
        "amount",
        "transaction_date",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_log(
    report: CleaningReport,
    path: Path,
) -> None:
    """Persist every transformation so cleaning remains traceable."""

    fieldnames = [
        "record_id",
        "field",
        "issue_type",
        "original_value",
        "cleaned_value",
        "action",
        "reason",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for issue in report.issues:
            writer.writerow(
                {
                    "record_id": issue.record_id,
                    "field": issue.field,
                    "issue_type": issue.issue_type,
                    "original_value": repr(issue.original_value),
                    "cleaned_value": repr(issue.cleaned_value),
                    "action": issue.action,
                    "reason": issue.reason,
                }
            )


# ---------------------------------------------------------------------------
# Advanced quality analysis
# ---------------------------------------------------------------------------

def compare_quality(
    original: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare measurable properties before and after cleaning."""

    def duplicate_count(rows: list[dict[str, Any]]) -> int:
        keys = [duplicate_key(row) for row in rows]
        return len(keys) - len(set(keys))

    return {
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "rows_removed": len(original) - len(cleaned),
        "original_duplicates": duplicate_count(original),
        "cleaned_duplicates": duplicate_count(cleaned),
        "original_missing_cells": sum(
            is_missing(value)
            for row in original
            for value in row.values()
        ),
        "cleaned_missing_cells": sum(
            is_missing(value)
            for row in cleaned
            for value in row.values()
        ),
    }


def print_cleaned_records(rows: list[dict[str, Any]]) -> None:
    """Display important cleaned fields without relying on external packages."""

    print("\nCLEANED RECORDS")
    print("=" * 72)

    for row in rows:
        print(
            f"{row['transaction_id']} | "
            f"{row['customer_name']} | "
            f"{row['country']} | "
            f"{row['state']} | "
            f"{row['category']} | "
            f"{row['amount']:.2f} | "
            f"{row['transaction_date'] or 'MISSING'}"
        )


def print_report(report: CleaningReport) -> None:
    """Display a concise audit summary."""

    print("\nCLEANING AUDIT")
    print("=" * 72)

    for issue_type, count in sorted(report.summary().items()):
        print(f"{issue_type:24} {count}")

    print("\nDetailed transformations:")
    for issue in report.issues:
        print(
            f"[{issue.issue_type}] "
            f"{issue.record_id}.{issue.field}: "
            f"{issue.action} | "
            f"{issue.original_value!r} -> {issue.cleaned_value!r}"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def clean_dataset(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], CleaningReport]:
    """
    Execute the cleaning pipeline in a deliberate order.

    Standardization happens before duplicate detection so that records with
    equivalent representations can be compared consistently. Missing-value
    handling and validation then operate on the normalized representation.
    """

    working = [dict(row) for row in rows]
    report = CleaningReport()

    standardize_inconsistent_values(working, report)
    working = remove_exact_business_duplicates(working, report)
    impute_missing_amounts(working, report)
    handle_missing_dates(working, report)

    # Outliers are identified after numeric normalization but are not silently
    # deleted. Statistical unusualness does not necessarily mean bad data.
    detect_iqr_outliers(working, "amount", report)

    errors = validate_clean_dataset(working)

    if errors:
        error_text = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Post-cleaning validation failed:\n{error_text}")

    return working, report


def main() -> None:
    """Run the complete demonstration."""

    original = build_sample_data()

    print("DATA CLEANING LABORATORY")
    print("=" * 72)
    print("The dataset intentionally contains missing, duplicate,")
    print("outlier, and inconsistent values.")

    profile = profile_dataset(original)
    print_profile(profile)

    cleaned, report = clean_dataset(original)

    comparison = compare_quality(original, cleaned)

    print("\nQUALITY COMPARISON")
    print("=" * 72)
    for key, value in comparison.items():
        print(f"{key:24} {value}")

    print_cleaned_records(cleaned)
    print_report(report)

    output_directory = Path("data_cleaning_output")
    output_directory.mkdir(exist_ok=True)

    cleaned_path = output_directory / "cleaned_transactions.csv"
    audit_path = output_directory / "cleaning_audit.csv"

    write_clean_csv(cleaned, cleaned_path)
    write_audit_log(report, audit_path)

    print("\nOUTPUT FILES")
    print("=" * 72)
    print(f"Cleaned dataset: {cleaned_path.resolve()}")
    print(f"Audit log:       {audit_path.resolve()}")

    print("\nKEY DESIGN DECISIONS")
    print("=" * 72)
    print(
        "Missing numeric amounts are imputed with a median, while missing "
        "dates remain explicitly missing because inventing a date can alter "
        "time-based analysis."
    )
    print(
        "Duplicates are removed using a business identity rather than a "
        "generic row-string comparison."
    )
    print(
        "Outliers are flagged instead of automatically deleted because an "
        "extreme transaction can be legitimate."
    )
    print(
        "Inconsistent categorical representations are converted using "
        "explicit domain mappings rather than unrestricted fuzzy matching."
    )


if __name__ == "__main__":
    main()
