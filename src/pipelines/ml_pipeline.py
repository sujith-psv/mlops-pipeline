from prefect import flow, task
import sys
import subprocess


# -----------------------------
# TASK: VALIDATE DATA
# -----------------------------

@task
def validate_data():

    print("Running data validation...")

    subprocess.run(
        [sys.executable, "src/validation/validate_data.py"],
        check=True
    )

    print("Data validation completed!")


# -----------------------------
# TASK: TRAIN MODEL
# -----------------------------

@task
def train_model():

    print("Training model...")

    subprocess.run(
        [sys.executable, "src/training/train.py"],
        check=True
    )

    print("Model training completed!")


# -----------------------------
# TASK: DETECT DRIFT
# -----------------------------

@task
def detect_drift():

    print("Running drift detection...")

    subprocess.run(
        [sys.executable, "src/monitoring/drift_detection.py"],
        check=True
    )

    print("Drift detection completed!")


# -----------------------------
# MAIN FLOW
# -----------------------------

@flow
def ml_pipeline():

    validate_data()

    train_model()

    detect_drift()

    print("Full ML pipeline completed successfully!")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":

    ml_pipeline()