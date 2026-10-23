"""Verify the released real-case decisions from derived scores and model knots.

No original discharge archive or pickle loading is required. This independent
check reproduces the final score-to-decision calculation; full reconstruction
of these real stations requires the provider observations and original masks.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path


def interpolate(value: float, xs: list[float], ys: list[float]) -> float:
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    right = bisect.bisect_right(xs, value)
    left = right - 1
    proportion = (value - xs[left]) / (xs[right] - xs[left])
    return ys[left] + proportion * (ys[right] - ys[left])


def verify(folder: Path) -> dict:
    model = json.loads((folder / "model_m_rf_fold1.json").read_text(encoding="utf-8"))
    with (folder / "diagnostic_features_and_expected_decisions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        cases = list(csv.DictReader(stream))
    with (folder / "historical_window_scores.csv").open(newline="", encoding="utf-8") as stream:
        histories = list(csv.DictReader(stream))
    xs = model["isotonic"]["x_thresholds"]
    ys = model["isotonic"]["y_thresholds"]
    assert len(xs) == len(ys) and all(a < b for a, b in zip(xs, xs[1:]))
    assert all(a <= b for a, b in zip(ys, ys[1:]))
    assert all(0 <= y <= 1 for y in ys)
    threshold = model["policy"]["threshold"]
    outcomes = []
    for case in cases:
        valid = [
            float(item["NSE"]) for item in histories
            if item["case_id"] == case["case_id"]
            and item["evaluable"].lower() == "true" and item["NSE"]
        ]
        assert len(valid) == int(case["hist_valid_count"])
        probability = None
        if valid:
            score = statistics.median(valid)
            assert math.isclose(score, float(case["hist_NSE_median"]), rel_tol=0, abs_tol=1e-12)
            probability = interpolate(score, xs, ys)
            assert math.isclose(probability, float(case["probability"]), rel_tol=0, abs_tol=1e-12)
            state = "adopt" if probability >= threshold else "reject_low_probability"
        else:
            state = "evidence_insufficient"
            assert not case["probability"] and not case["hist_NSE_median"]
        assert math.isclose(threshold, float(case["threshold"]), rel_tol=0, abs_tol=1e-12)
        assert state == case["state"], (case["case_id"], state, case["state"])
        outcomes.append({"case_id": case["case_id"], "state": state, "probability": probability})
    assert len(outcomes) == 4
    assert {item["state"] for item in outcomes} == {
        "adopt", "reject_low_probability", "evidence_insufficient"
    }
    return {"status": "PASS", "case_count": len(outcomes),
            "verification_scope": "historical score summary, frozen interpolation and three-state decisions",
            "original_discharge_read": False, "cases": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
