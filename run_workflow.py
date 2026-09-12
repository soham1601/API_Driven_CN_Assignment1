"""
main.py
Entry point that runs the full pipeline end-to-end:
  1. Ingest data
  2. Preprocess (stats, missing values, imputation) -- kept un-scaled for EDA
  3. Run EDA (correlation, binning, encoding, feature importance, charts)
  4. Normalize (separately, to demonstrate the requirement for the Word doc)

Run this locally first to make sure everything works end-to-end before
wrapping load_data / preprocess_pipeline / run_eda as @task functions
inside a Prefect @flow for the DataOps (automation + scheduling) part.
"""

import logging
from data_ingest import load_data, get_basic_info
from data_preprocess import preprocess_pipeline, normalize_numeric_columns
from data_EDA import run_eda

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(path=None) -> None:
    logger.info("=== STEP 1: Data Ingestion ===")
    df = load_data(path)   # None -> falls back to Dataset/hotel_bookings.csv (or DATA_PATH env var)
    get_basic_info(df)

    logger.info("=== STEP 2: Preprocessing (stats, missing values, imputation) ===")
    df_clean = preprocess_pipeline(df)

    logger.info("=== STEP 3: Exploratory Data Analysis ===")
    eda_results = run_eda(df_clean)
    print("Top correlated pairs / feature importances are saved in ./eda_outputs/")

    logger.info("=== STEP 4: Normalization (for documentation) ===")
    df_normalized = normalize_numeric_columns(df_clean)
    print(df_normalized.select_dtypes(include="number").head())

    logger.info("=== Pipeline run complete ===")


if __name__ == "__main__":
    run_pipeline()