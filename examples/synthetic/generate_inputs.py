"""Create deterministic artificial monthly inputs, unrelated to GSIM records.

The separate validation_truth.csv must never be supplied to reconstruction or
diagnostic fitting. It exists only for a post-decision software demonstration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261023)
    count = 252
    time = np.arange(count)
    dates = pd.date_range("1995-01-01", periods=count, freq="MS")
    latent = np.zeros(count)
    for index in range(1, count):
        latent[index] = 0.72 * latent[index - 1] + rng.normal(0, 0.45)
    seasonal = np.sin(2 * np.pi * time / 12) + 0.2 * np.cos(4 * np.pi * time / 12)
    target = np.maximum(1.0, 25 + 8 * seasonal + 4 * latent + rng.normal(0, 0.4, count))
    values = {"date": dates.strftime("%Y-%m"), "target": target.copy()}
    for donor in range(1, 9):
        series = np.maximum(
            0.1,
            (0.65 + 0.1 * donor) * (24 + 7.8 * seasonal + 4 * latent)
            + rng.normal(0, 0.4 + 0.08 * donor, count),
        )
        series[rng.random(count) < 0.03] = np.nan
        values[f"donor_{donor}"] = series
    frame = pd.DataFrame(values)
    gap = (time >= 108) & (time < 132)
    frame.loc[gap, "target"] = np.nan
    frame.to_csv(output / "monthly_flow.csv", index=False, float_format="%.12g")
    sparse = frame.copy()
    sparse.loc[(time % 3) != 0, "target"] = np.nan
    sparse.to_csv(output / "monthly_flow_sparse.csv", index=False, float_format="%.12g")
    metadata = [{"station_id": "target", "latitude": 45.0, "longitude": 10.0}]
    metadata += [
        {"station_id": f"donor_{index}", "latitude": 45 + 0.03 * index,
         "longitude": 10 + 0.04 * index}
        for index in range(1, 9)
    ]
    pd.DataFrame(metadata).to_csv(output / "station_metadata.csv", index=False)
    pd.DataFrame({"date": dates[gap].strftime("%Y-%m"), "target_truth": target[gap]}).to_csv(
        output / "validation_truth.csv", index=False, float_format="%.12g"
    )
    manifest = {
        "data_kind": "synthetic",
        "derived_from_GSIM": False,
        "seed": 20261023,
        "calendar_start": "1995-01",
        "calendar_end": "2015-12",
        "gap_start": "2004-01",
        "gap_end": "2005-12",
        "gap_end_is_inclusive": True,
        "gap_position": "interior",
        "units": "arbitrary synthetic flow units",
        "target_column": "target",
        "donor_columns": [f"donor_{index}" for index in range(1, 9)],
        "truth_usage": "Post-decision validation only; never a reconstruction or diagnostic input.",
        "sparse_variant": "Every third outside-gap target month remains visible, so no fully observed 24-month historical window exists.",
        "probability": "A separately trained and calibrated model is required. No success probability is invented for this synthetic input.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "input")
    generate(parser.parse_args().output)
