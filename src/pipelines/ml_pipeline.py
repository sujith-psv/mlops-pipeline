from prefect import flow, task
import sys
import subprocess


# ============================================================
# TASK: DATA VALIDATION
# ============================================================

@task(
    name="Data Validation",
    retries=1,
    retry_delay_seconds=5
)
def validate_data():

    print("\n" + "=" * 60)
    print("TASK: DATA VALIDATION")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable,
            "src/validation/validate_data.py"
        ],
        check=True
    )

    print("Data validation completed successfully!")

    return "validation_passed"


# ============================================================
# TASK: MODEL TRAINING
# ============================================================

@task(
    name="Model Training",
    retries=1,
    retry_delay_seconds=10
)
def train_model(validation_result):

    print("\n" + "=" * 60)
    print("TASK: MODEL TRAINING")
    print("=" * 60)

    if validation_result != "validation_passed":
        raise RuntimeError(
            "Training blocked because data validation failed."
        )

    subprocess.run(
        [
            sys.executable,
            "src/training/train.py"
        ],
        check=True
    )

    print("Model training completed successfully!")

    return "training_completed"


# ============================================================
# TASK: DRIFT DETECTION
# ============================================================

@task(
    name="Drift Detection",
    retries=1,
    retry_delay_seconds=5
)
def detect_drift(training_result):

    print("\n" + "=" * 60)
    print("TASK: DRIFT DETECTION")
    print("=" * 60)

    if training_result != "training_completed":
        raise RuntimeError(
            "Drift detection blocked because training failed."
        )

    subprocess.run(
        [
            sys.executable,
            "src/monitoring/drift_detection.py"
        ],
        check=True
    )

    print("Drift detection completed successfully!")

    return "pipeline_completed"


# ============================================================
# MAIN PREFECT FLOW
# ============================================================

@flow(
    name="Telco Churn MLOps Pipeline"
)
def ml_pipeline():

    validation_result = validate_data()

    training_result = train_model(
        validation_result
    )

    pipeline_result = detect_drift(
        training_result
    )

    print("\n" + "=" * 60)
    print("FULL ML PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return pipeline_result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ml_pipeline()