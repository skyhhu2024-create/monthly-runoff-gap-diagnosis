"""Run the two real monthly-input cases and check their released reference outputs.

Install the repository package first. Run from any working directory, e.g.:
python examples/real_cases/reproduce.py --output outputs/real_cases
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = root / "model_m_rf_fold1.json"
    metadata = json.loads((root / "case_metadata.json").read_text(encoding="utf-8"))
    checks = []
    for station_id in ("BR_0001891", "US_0002700"):
        case = next(item for item in metadata if item["station_id"] == station_id)
        folder = root / station_id
        destination = output / station_id
        command = [
            sys.executable, "-m", "gsim_gap_diagnostic", "reconstruct",
            "--flow", str(folder / "monthly_flow.csv"),
            "--metadata", str(folder / "station_metadata.csv"),
            "--gap-start", case["gap_start_date"], "--gap-end", case["gap_end_date"],
            "--case-id", case["case_id"], "--station-id", station_id,
            "--algorithm", "rf", "--position", "interior", "--seed", "20261014",
            "--model", str(model), "--output", str(destination),
        ]
        subprocess.run(command, check=True)
        expected = folder / "expected_output"
        read = lambda file: pd.read_csv(file, float_precision="round_trip")
        feature_columns = ["hist_window_count", "hist_valid_count", "hist_R_median",
                           "hist_R_iqr", "hist_NSE_median", "hist_NSE_iqr", "hist_joint_rate"]
        actual_features = read(destination / "visible_features.csv")
        expected_features = read(expected / "visible_features.csv")
        assert np.allclose(actual_features[feature_columns], expected_features[feature_columns],
                           rtol=0, atol=1e-12, equal_nan=True), station_id + " historical features differ"
        actual_history = read(destination / "history_records.csv")
        expected_history = read(expected / "history_records.csv")
        assert list(actual_history.start) == list(expected_history.start)
        assert np.allclose(actual_history[["R", "NSE"]], expected_history[["R", "NSE"]],
                           rtol=0, atol=1e-12, equal_nan=True), station_id + " trial scores differ"
        actual = read(destination / "decisions.csv")
        reference = read(expected / "decisions.csv")
        assert list(actual.decision) == list(reference.decision)
        assert np.allclose(actual.probability, reference.probability, rtol=0, atol=1e-12, equal_nan=True)
        prediction = read(destination / "current_reconstruction.csv")
        prediction_reference = read(expected / "current_reconstruction.csv")
        assert list(prediction.date) == list(prediction_reference.date)
        assert np.allclose(prediction.prediction, prediction_reference.prediction,
                           rtol=0, atol=1e-10, equal_nan=True), station_id + " reconstruction differs"
        checks.append({"station_id": station_id, "status": "PASS",
                       "historical_trials": len(actual_history),
                       "probability": float(actual.probability.iloc[0]),
                       "decision": actual.decision.iloc[0]})
    report = {"status": "PASS", "cases": checks,
              "truth_read_by_this_runner": False,
              "scope": "Reconstruction, historical windows, trial scores, M probability and decision."}
    (output / "reproduction_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
