import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

reference_data = pd.read_csv(
    "data/raw/telco.csv"
)


# ============================================================
# CREATE CURRENT DATA
# ============================================================

current_data = reference_data.copy()


# Simulate production drift
current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.5
)

current_data["tenure"] = (
    current_data["tenure"] + 10
)


# ============================================================
# CREATE EVIDENTLY REPORT
# ============================================================

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ============================================================
# SAVE HTML REPORT
# ============================================================

os.makedirs("reports", exist_ok=True)

report.save_html(
    "reports/drift_report.html"
)


# ============================================================
# EXTRACT DRIFT INFORMATION
# ============================================================

report_dict = report.as_dict()

drifted_columns = None
total_columns = len(reference_data.columns)


def find_drift_result(obj):
    """
    Search the Evidently report recursively for
    number_of_drifted_columns / share_of_drifted_columns.
    """

    global drifted_columns

    if isinstance(obj, dict):

        if "number_of_drifted_columns" in obj:
            drifted_columns = obj["number_of_drifted_columns"]
            return

        for value in obj.values():
            find_drift_result(value)

            if drifted_columns is not None:
                return

    elif isinstance(obj, list):

        for item in obj:
            find_drift_result(item)

            if drifted_columns is not None:
                return


find_drift_result(report_dict)


# ============================================================
# MONITORING DECISION
# ============================================================

print("\n" + "=" * 60)
print("DATA DRIFT MONITORING")
print("=" * 60)

print("\nReference dataset:")
print(reference_data.shape)

print("\nCurrent dataset:")
print(current_data.shape)

print("\nDrift simulation:")
print("MonthlyCharges: +50%")
print("tenure: +10")


if drifted_columns is not None:

    drift_percentage = (
        drifted_columns / total_columns
    ) * 100

    print("\nDrifted columns:")
    print(
        f"{drifted_columns} / "
        f"{total_columns}"
    )

    print(
        "Drift percentage:",
        f"{drift_percentage:.2f}%"
    )

    # Engineering monitoring threshold
    DRIFT_THRESHOLD = 10.0

    print(
        "Monitoring threshold:",
        f"{DRIFT_THRESHOLD:.2f}%"
    )

    if drift_percentage >= DRIFT_THRESHOLD:

        print("\nWARNING: DATA DRIFT DETECTED")
        print("Model monitoring recommends investigation.")

    else:

        print("\nDATA DRIFT BELOW MONITORING THRESHOLD")
        print("No retraining action required.")


else:

    print(
        "\nCould not extract drift statistics "
        "from Evidently report."
    )


# ============================================================
# REPORT
# ============================================================

print("\nDrift report generated successfully!")
print("Report: reports/drift_report.html")