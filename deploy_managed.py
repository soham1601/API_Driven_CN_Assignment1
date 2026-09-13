"""
deploy_managed.py
Registers the hotel-booking-eda-pipeline flow as a Prefect Cloud deployment
using MANAGED execution — Prefect runs it on their own infrastructure,
so no local worker or VM is needed.

Since managed execution pulls code from a remote source (GitHub) rather
than your local disk, this points at the pushed repo instead of the local
run_workflow.py file directly.

This does NOT execute the pipeline itself — it just tells Prefect Cloud
where to find the code and how often to run it. Prefect Cloud then runs
it automatically every 2 minutes, with no worker process needed anywhere.
"""

from prefect import flow

# TODO: update this to your actual GitHub repo URL
GITHUB_REPO_URL = "https://github.com/mathurd96/API_Driven_CN_Assignment1.git"

if __name__ == "__main__":
    flow.from_source(
        source=GITHUB_REPO_URL,
        entrypoint="run_workflow.py:run_pipeline",
    ).deploy(
        name="hotel-eda-deployment",
        work_pool_name="assignment1-managed-pool",
        interval=120,  # every 2 minutes, matching the assignment requirement
        job_variables={
            "pip_packages": [
                "pandas",
                "numpy",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "requests",
            ]
        },
    )
    print("Deployment registered. Check https://app.prefect.cloud for scheduled runs.")
