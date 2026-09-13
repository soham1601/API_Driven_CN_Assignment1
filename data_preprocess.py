"""
data_preprocess.py
Responsible for Activity 1.3: summary statistics, missing-value handling,
dtype inspection, and normalization.

All of this ALSO gets written to a plain-text report file (see
write_preprocessing_report below) so there is something you can open once
and screenshot cleanly for the Word document, instead of scrolling back
through terminal output that may have already scrolled past.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def show_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Displays summary stats for all columns (numeric + categorical)."""
    summary = df.describe(include="all")
    logger.info("Generated summary statistics")
    print(summary)
    return summary


def show_data_types(df: pd.DataFrame) -> pd.Series:
    """Displays the dtype of every column."""
    dtypes = df.dtypes
    logger.info("Retrieved column data types")
    print(dtypes)
    return dtypes


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Returns a count of missing values per column."""
    missing = df.isnull().sum()
    logger.info(f"Missing value check complete. Columns with nulls: "
                f"{missing[missing > 0].to_dict()}")
    print(missing[missing > 0])
    return missing


def impute_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing numeric values — but NOT with the same strategy for every
    column. 'agent' and 'company' are ID codes, not real numeric quantities,
    so a NaN there means "no agent/company involved" (a direct booking) —
    filling that with the median agent ID would invent a fake, meaningless
    agent. Everything else gets standard median imputation.
    """
    df = df.copy()
    id_like_cols = ["agent", "company"]  # NaN here means "not applicable", not "unknown"

    for col in id_like_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
            logger.info(f"Imputed ID-like column '{col}' missing values with 0 (= not applicable)")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if col in id_like_cols:
            continue
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed '{col}' missing values with median={median_val}")
    return df


def impute_missing_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing categorical/object values with the column mode, or 'Unknown' if no mode exists."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Imputed categorical '{col}' missing values with '{fill_val}'")
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Counts exact duplicate rows. This dataset (Hotel Booking Demand) is
    widely known to contain a large number of exact duplicates — worth
    checking and reporting explicitly rather than assuming clean data.
    """
    dup_count = df.duplicated().sum()
    pct = round(100 * dup_count / len(df), 2)
    logger.info(f"Found {dup_count} exact duplicate rows ({pct}% of the dataset)")
    return dup_count


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df


def flag_invalid_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags (does not silently drop) bookings with zero total guests
    (adults + children + babies == 0) — another known data-quality issue
    in this dataset, since a booking with no guests at all is logically
    invalid. Adds an 'is_valid_booking' column so you can decide in your
    report whether to filter these out, rather than removing data silently.
    """
    df = df.copy()
    guest_cols = [c for c in ["adults", "children", "babies"] if c in df.columns]
    if guest_cols:
        total_guests = df[guest_cols].sum(axis=1)
        df["is_valid_booking"] = (total_guests > 0).astype(int)
        invalid_count = (df["is_valid_booking"] == 0).sum()
        logger.info(f"Flagged {invalid_count} bookings with zero total guests "
                    f"(adults+children+babies == 0) via 'is_valid_booking' column")
    return df


def normalize_numeric_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Min-Max scales numeric columns to [0, 1].
    Pass `columns` explicitly if you only want to scale a subset
    (e.g. exclude the target column 'is_canceled').
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if "is_canceled" in columns:
            columns.remove("is_canceled")  # don't scale the target

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"Normalized columns: {columns}")
    return df


def write_preprocessing_report(shape_before, shape_after, summary, dtypes, missing,
                                dup_count, dup_pct, invalid_count) -> None:
    """
    Writes one consolidated, human-readable text file covering everything
    checked during preprocessing: shape, summary statistics, data types,
    missing values, duplicates found/removed, and invalid bookings flagged.
    Open this file once and screenshot it — this is the cleanest evidence
    for the Word document, rather than relying on terminal scrollback.
    """
    lines = []
    lines.append("PREPROCESSING SUMMARY REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Shape before preprocessing: {shape_before[0]} rows, {shape_before[1]} columns")
    lines.append(f"Shape after preprocessing:  {shape_after[0]} rows, {shape_after[1]} columns")
    lines.append("")

    lines.append("SUMMARY STATISTICS (transposed: one row per column, for readability)")
    lines.append("-" * 50)
    lines.append(summary.transpose().to_string())
    lines.append("")

    lines.append("COLUMN DATA TYPES")
    lines.append("-" * 50)
    lines.append(dtypes.to_string())
    lines.append("")

    lines.append("MISSING VALUES (before imputation)")
    lines.append("-" * 50)
    missing_nonzero = missing[missing > 0]
    if missing_nonzero.empty:
        lines.append("No missing values found.")
    else:
        lines.append(missing_nonzero.to_string())
    lines.append("")

    lines.append("DUPLICATE ROWS")
    lines.append("-" * 50)
    lines.append(f"Found {dup_count} exact duplicate rows ({dup_pct}% of the dataset).")
    lines.append(f"These were removed before any further analysis, to avoid the "
                  f"train/test leakage issue described in our report.")
    lines.append("")

    lines.append("INVALID BOOKINGS")
    lines.append("-" * 50)
    lines.append(f"Flagged {invalid_count} bookings with zero total guests "
                  f"(adults + children + babies == 0) via the 'is_valid_booking' column.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(REPORTS_DIR / "preprocessing_report.txt", "w") as f:
        f.write(report_text)

    logger.info("Preprocessing report saved to reports/preprocessing_report.txt")


def preprocess_pipeline(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Orchestrator: runs stats/dtype/missing-value/duplicate checks and
    imputation, and returns a CLEAN but NOT YET NORMALIZED DataFrame.

    Deliberately does not scale here — EDA (binning thresholds, chart axis
    labels, correlation readability) is easier to interpret on raw values.
    Call normalize_numeric_columns() separately, after EDA, purely to
    demonstrate the normalization requirement for the Word doc.
    """
    shape_before = df.shape

    summary = show_summary_statistics(df)
    dtypes = show_data_types(df)
    missing = check_missing_values(df)
    dup_count = check_duplicates(df)
    dup_pct = round(100 * dup_count / len(df), 2)

    if drop_duplicates:
        df = remove_duplicates(df)
    df = impute_missing_numeric(df)
    df = impute_missing_categorical(df)
    df = flag_invalid_bookings(df)

    invalid_count = int((df["is_valid_booking"] == 0).sum()) if "is_valid_booking" in df.columns else 0
    shape_after = df.shape

    write_preprocessing_report(shape_before, shape_after, summary, dtypes, missing,
                                dup_count, dup_pct, invalid_count)

    logger.info("Preprocessing (clean/impute) complete — call normalize_numeric_columns() separately")
    return df


if __name__ == "__main__":
    from data_ingest import load_data
    df = load_data()
    df_clean = preprocess_pipeline(df)
    df_normalized = normalize_numeric_columns(df_clean)
    print("Before normalization:\n", df_clean.select_dtypes(include="number").head())
    print("After normalization:\n", df_normalized.select_dtypes(include="number").head())