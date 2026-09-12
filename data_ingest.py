"""
data_ingest.py
Responsible for loading the raw dataset from disk (Sub-Objective 1, Activity 1.2).
"""

import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Path relative to THIS FILE's location, not to whatever folder you happen to
# run the script from. Works the same on your laptop, a teammate's machine,
# a GCP VM, or inside a Prefect worker — no hardcoded absolute paths.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "Dataset" / "hotel_bookings.csv"

# Optional override: set an environment variable DATA_PATH if you ever need
# to point at a different location without touching the code, e.g. on a VM:
#   export DATA_PATH=/home/ubuntu/assignment1/Dataset/hotel_bookings.csv
DATA_PATH = Path(os.environ.get("DATA_PATH", DEFAULT_DATA_PATH))


def load_data(path: Path = None) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame and log basic ingestion facts.
    Falls back to DATA_PATH (Dataset/hotel_bookings.csv relative to this
    file, or the DATA_PATH env var if set) when no path is given.
    """
    path = Path(path) if path else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            f"Make sure the CSV is inside a 'Dataset' folder next to this script, "
            f"or set the DATA_PATH environment variable."
        )

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset from '{path}' with shape {df.shape}")
    if df.shape[0] < 10000:
        logger.warning("Dataset has fewer than 10,000 rows — assignment requires >= 10,000.")
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    """
    Print/log a quick snapshot of the ingested data — useful for the
    'display application/data details' part of your Word doc.
    """
    logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    logger.info(f"Column names: {list(df.columns)}")
    print(df.head())


if __name__ == "__main__":
    df = load_data()
    get_basic_info(df)