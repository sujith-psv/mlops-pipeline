import pandas as pd
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# -----------------------------
# LOAD REFERENCE DATA
# -----------------------------

reference_data = pd.read_csv("data/raw/telco.csv")


# -----------------------------
# CREATE CURRENT DATA
# -----------------------------

current_data = reference_data.copy()


# Simulate drift
current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.5
)

current_data["tenure"] = (
    current_data["tenure"] + 10
)


# -----------------------------
# CREATE DRIFT REPORT
# -----------------------------

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)


report.run(
    reference_data=reference_data,
    current_data=current_data
)


# -----------------------------
# SAVE HTML REPORT
# -----------------------------

report.save_html("reports/drift_report.html")
print("Drift report generated successfully!")
#python src/monitoring/drift_detection.py