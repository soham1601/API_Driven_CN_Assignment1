"""
data_EDA.py
Responsible for Activity 1.4: correlation, binning, encoding,
feature importance, and univariate/bivariate visualizations.
All charts are saved to ./eda_outputs/ so they're ready to paste into the Word doc.
Model evaluation metrics (accuracy/precision/recall/confusion matrix) are
written separately to ./reports/, kept apart from the charts folder.
"""

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Relative to this file's location, not the current working directory —
# so charts always land in the same place regardless of where you run from.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Columns like agent/company are identifier codes, not real numeric
# quantities. Tree-based models (RandomForest) are biased toward ranking
# high-cardinality ID columns as "important" even when they carry no real
# business signal, so they're excluded from feature importance/modeling.
ID_LIKE_COLS = ["agent", "company"]


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and plots a correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

    logger.info(f"Correlation matrix computed for {corr.shape[0]} numeric columns")
    return corr


def bin_feature(df: pd.DataFrame, column: str, bins: list, labels: list,
                 new_column: str = None) -> pd.DataFrame:
    """Bins a continuous numeric column into categorical ranges."""
    df = df.copy()
    new_column = new_column or f"{column}_bin"
    df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
    logger.info(f"Binned '{column}' into '{new_column}' with labels {labels}")
    return df


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encodes the given categorical columns."""
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    logger.info(f"One-hot encoded columns: {columns}")
    return df_encoded


def encode_binary_target(y: pd.Series) -> pd.Series:
    """
    Ensures the target is numeric 0/1 for sklearn metrics that default to
    pos_label=1 (precision_score, recall_score, f1_score). Some datasets
    (including this one) store the target as strings like 'yes'/'no'
    rather than 1/0 — this handles that transparently.

    Uses pd.api.types.is_numeric_dtype() rather than checking for a
    specific dtype name, since pandas versions differ on how they label
    string columns (older pandas: 'object', pandas 2.x+/3.x with the new
    string backend: 'str' or 'string') — checking "is it numeric?" is
    robust across all of them.
    """
    if not pd.api.types.is_numeric_dtype(y):
        unique_vals = sorted(y.dropna().unique())
        # Common case: yes/no -> 1/0. Otherwise fall back to alphabetical order.
        mapping = {"no": 0, "yes": 1} if set(unique_vals) <= {"no", "yes"} \
            else {val: i for i, val in enumerate(unique_vals)}
        logger.info(f"Target column was non-numeric ({unique_vals}); encoded using mapping {mapping}")
        return y.map(mapping)
    return y


def feature_importance(df_encoded: pd.DataFrame, target_col: str,
                        drop_cols: list = None, top_n: int = 10) -> pd.Series:
    """
    Trains a quick RandomForest and plots the top feature importances.
    ID-like columns (agent, company) are excluded by default — see
    ID_LIKE_COLS note above for why they'd otherwise distort the ranking.
    """
    drop_cols = (drop_cols or []) + ID_LIKE_COLS
    y = encode_binary_target(df_encoded[target_col])
    X = df_encoded.drop(columns=[target_col] + drop_cols, errors="ignore")
    X = X.select_dtypes(include="number").fillna(0)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns) \
                    .sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

    logger.info(f"Feature importance computed (excluding ID-like columns {ID_LIKE_COLS}). "
                f"Top feature: {importances.index[0]}")
    return importances


def evaluate_model(df_encoded: pd.DataFrame, target_col: str,
                    drop_cols: list = None, test_size: float = 0.2) -> dict:
    """
    Trains/evaluates a RandomForest on a proper train/test split and writes
    a full metrics report (accuracy, precision, recall, F1, confusion matrix)
    to ./reports/ — kept separate from the EDA charts folder.
    """
    drop_cols = (drop_cols or []) + ID_LIKE_COLS
    y = encode_binary_target(df_encoded[target_col])
    X = df_encoded.drop(columns=[target_col] + drop_cols, errors="ignore")
    X = X.select_dtypes(include="number").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- Confusion matrix chart (goes in eda_outputs/, since it IS a chart) ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Canceled", "Canceled"],
                yticklabels=["Not Canceled", "Canceled"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    # --- Text/JSON metrics report (goes in reports/, separate from charts) ---
    with open(REPORTS_DIR / "metrics_report.txt", "w") as f:
        f.write("MODEL EVALUATION METRICS\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFull classification report:\n")
        f.write(report_text)
        f.write("\nConfusion matrix (rows=actual, cols=predicted):\n")
        f.write(str(cm))

    with open(REPORTS_DIR / "metrics_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model evaluation complete: accuracy={metrics['accuracy']}, "
                f"precision={metrics['precision']}, recall={metrics['recall']}, "
                f"f1={metrics['f1_score']}. Full report saved to reports/metrics_report.txt")
    return metrics


def univariate_analysis(df: pd.DataFrame, column: str, clip_upper_quantile: float = 0.99) -> None:
    """
    Plots the distribution of a single numeric column.
    Extreme outliers (e.g. a data-entry error like adr > 5000) can crush
    an entire histogram into one bar. `clip_upper_quantile` zooms the
    x-axis into the 0-99th percentile range for a readable chart — this
    only affects the PLOT, the underlying data/df is untouched.
    """
    upper_bound = df[column].quantile(clip_upper_quantile)
    outlier_count = (df[column] > upper_bound).sum()

    plt.figure(figsize=(8, 5))
    df[df[column] <= upper_bound][column].hist(bins=30)
    plt.title(f"Distribution of {column} (clipped at {int(clip_upper_quantile*100)}th percentile)")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"univariate_{column}.png")
    plt.close()
    logger.info(f"Univariate chart saved for '{column}' "
                f"({outlier_count} outlier(s) above {upper_bound:.1f} excluded from the plot only)")


def bivariate_analysis(df: pd.DataFrame, categorical_col: str, target_col: str) -> None:
    """Plots a categorical feature against the target (count/hue plot)."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=categorical_col, hue=target_col, data=df)
    plt.xticks(rotation=90)
    plt.title(f"{categorical_col} vs {target_col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"bivariate_{categorical_col}_vs_{target_col}.png")
    plt.close()
    logger.info(f"Bivariate chart saved for '{categorical_col}' vs '{target_col}'")


def run_eda(df: pd.DataFrame) -> dict:
    """
    Orchestrator: runs the full EDA sequence and returns a dict of results.
    This is the single function main.py (or a Prefect task) should call.
    """
    results = {}

    results["correlation"] = correlation_analysis(df)

    df = bin_feature(df, column="lead_time",
                      bins=[-1, 7, 30, 90, 365, 1000],
                      labels=["last-minute", "short", "medium", "long", "very-long"])

    # Engineered feature: does the guest end up in a different room type than
    # they reserved? Room reassignment is a plausible cancellation signal
    # and is worth testing against the raw day/week-number columns below.
    if {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
        df["room_type_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
        logger.info("Engineered 'room_type_mismatch' feature "
                    f"({df['room_type_mismatch'].sum()} bookings had a mismatch)")

    df_encoded = encode_categorical(df, columns=[
        "hotel", "meal", "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "arrival_date_month",
        "reserved_room_type", "assigned_room_type"
    ])

    # NOTE on cardinality bias: 'arrival_date_day_of_month' (31 unique values)
    # and 'arrival_date_week_number' (~53 unique values) previously ranked
    # suspiciously high in feature importance — RandomForest importance is
    # biased toward high-cardinality numeric columns even when the real
    # signal is just "which season/month is it". 'arrival_date_month' above
    # (only 12 categories) is included so you can compare its importance
    # against the raw day/week columns in your report — if month alone
    # captures most of the same signal, that's good evidence the raw
    # day/week importance was partly a cardinality artifact.
    results["feature_importance"] = feature_importance(
        df_encoded,
        target_col="is_canceled",
        drop_cols=["lead_time_bin", "reservation_status", "reservation_status_date"]
    )

    results["metrics"] = evaluate_model(
        df_encoded,
        target_col="is_canceled",
        drop_cols=["lead_time_bin", "reservation_status", "reservation_status_date"]
    )

    univariate_analysis(df, column="adr")
    bivariate_analysis(df, categorical_col="market_segment", target_col="is_canceled")

    logger.info("EDA pipeline complete. Charts saved to ./eda_outputs/, metrics saved to ./reports/")
    return results


if __name__ == "__main__":
    from data_ingest import load_data
    from data_preprocess import preprocess_pipeline

    df = load_data()
    df_clean = preprocess_pipeline(df)
    run_eda(df_clean)