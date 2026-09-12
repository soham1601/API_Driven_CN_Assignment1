"""
run_workflow.py
Prefect-native entry point for Sub-Objective 1 (DataOps).

This does NOT duplicate any pipeline logic — it wraps the already-tested
functions from data_ingest.py, data_preprocess.py, and data_EDA.py as
Prefect @task functions inside a single @flow, so the exact same code you
already validated locally can now be scheduled, monitored, and retried via
Prefect Cloud.

Deploy with:
    prefect deploy run_workflow.py:run_pipeline -n hotel-eda-deployment --interval 120
    prefect worker start --pool "default-agent-pool"

Run locally (without Prefect scheduling, just to sanity-check the flow):
    python run_workflow.py
"""

from prefect import flow, task, get_run_logger

from data_ingest import load_data, get_basic_info
from data_preprocess import preprocess_pipeline, normalize_numeric_columns
from data_EDA import run_eda


@task(name="ingest-data", retries=2, retry_delay_seconds=10)
def ingest_task():
    """Wraps data_ingest.load_data() — retries twice on transient I/O errors."""
    logger = get_run_logger()
    df = load_data()
    logger.info(f"Ingested dataset with shape {df.shape}")
    get_basic_info(df)
    return df


@task(name="preprocess-data")
def preprocess_task(df):
    """Wraps data_preprocess.preprocess_pipeline() — stats, missing values, duplicates, imputation."""
    logger = get_run_logger()
    df_clean = preprocess_pipeline(df)
    logger.info(f"Preprocessing complete. Cleaned shape: {df_clean.shape}")
    return df_clean


@task(name="run-eda")
def eda_task(df_clean):
    """Wraps data_EDA.run_eda() — correlation, binning, encoding, feature importance, model metrics, charts."""
    logger = get_run_logger()
    results = run_eda(df_clean)
    logger.info(f"EDA complete. Model metrics: {results.get('metrics')}")
    return results


@task(name="normalize-data")
def normalize_task(df_clean):
    """Wraps data_preprocess.normalize_numeric_columns() — run separately, after EDA, for documentation."""
    logger = get_run_logger()
    df_normalized = normalize_numeric_columns(df_clean)
    logger.info("Normalization complete (numeric columns scaled to 0-1)")
    return df_normalized


@flow(name="hotel-booking-eda-pipeline", log_prints=True)
def run_pipeline():
    """
    Orchestrates the full pipeline as a Prefect flow:
    ingest -> preprocess -> EDA (charts + metrics) -> normalize (for docs).
    Each @task above shows up as its own node in the Prefect Cloud flow-run
    graph, with its own logs, duration, and retry history.
    """
    logger = get_run_logger()
    logger.info("=== Starting hotel-booking-eda-pipeline run ===")

    df = ingest_task()
    df_clean = preprocess_task(df)
    eda_results = eda_task(df_clean)
    df_normalized = normalize_task(df_clean)

    logger.info("=== Pipeline run complete ===")
    return {"eda_results": eda_results}


if __name__ == "__main__":
    run_pipeline()