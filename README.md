# Monthly streamflow gap recoverability diagnosis

Diagnose an individual monthly streamflow gap from information available outside
that gap. A chosen reconstruction algorithm is tested on historical pseudo-gaps
within the same station. Historical performance is mapped to an empirical
success probability, supporting three decisions: accept, reject for low
probability, or abstain because historical evidence is insufficient.

This repository implements the methods in **Diagnosing the recoverability of
monthly streamflow gaps using within-station pseudo-gaps** by Mingrui Chen,
Bin Yong, Weihong Liao, Haichen Li, Wei Dai, Jia Wang, and Hao Wang.

## Install

Use Python 3.12 or newer. From a fresh environment:

```shell
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
gsim-gap --help
```

`requirements.txt` records the versions used for release verification. XGBoost
is used by the fifth conventional algorithm. Optional LSTM/TCN implementations
from the supplementary analysis require PyTorch (`pip install -e ".[neural]"`).

## Run the paper's real examples

```shell
python examples/real_cases/verify_expected.py
python examples/real_cases/reproduce.py --output outputs/real_cases
```

The examples include the BR_0001891 artificial gap and US_0002700 real internal
gap shown in the manuscript. Both contain the actual monthly MEAN inputs,
candidate donors, historical-window results and expected decisions. Two further
derived examples illustrate low-probability rejection and insufficient evidence.
See [the case instructions](examples/real_cases/README.md) for the complete
reconstruction command and [data attribution](examples/real_cases/ATTRIBUTION.md).

For an independently generated input example:

```shell
python examples/synthetic/generate_inputs.py
gsim-gap reconstruct --flow examples/synthetic/input/monthly_flow.csv --metadata examples/synthetic/input/station_metadata.csv --gap-start 2004-01 --gap-end 2005-12 --case-id synthetic_demo --algorithm rf --output outputs/synthetic
```

The synthetic data demonstrate execution. They are explicitly separate from
the real GSIM examples and do not establish empirical diagnostic performance.
Supply a fitted, compatible diagnostic model with `--model` to obtain a decision.

## Apply and fit diagnosis

```shell
gsim-gap diagnose --features examples/real_cases/visible_features.csv --model examples/real_cases/model_m_rf_fold1.json --output outputs/real_decisions.csv
gsim-gap fit --features features_with_train_cal_roles.csv --labels historical_training_labels.csv --method M --output outputs/fitted_M
```

M uses median historical NSE and an increasing isotonic mapping fitted in TRAIN;
CAL selects the acceptance threshold. H uses six historical summaries with a
spline logistic model and CAL probability calibration. Target observations
inside the current query must be missing before reconstruction. Current query
labels are joined only after features and diagnostic predictions are stored.
The input schema is described in [the data contract](docs/data_contract.md).

## Reproduce the complete study results

Unpack the accompanying data release, then run:

```shell
python scripts/replay_published_results.py --data-root ../zenodo_data --output outputs/replay
python scripts/replay_published_results.py --data-root ../zenodo_data --output outputs/refit --refit
python scripts/reproduce_reconstructions.py --help
```

The first command replays all 81,520 primary diagnoses: 8152 registered gaps ×
five algorithms × two use requirements. The second additionally refits TRAIN
maps and CAL policies using the original development cohort. The reconstruction
entry point rebuilds selected registered cases from the archived monthly inputs.
The data release README maps every result group and table to its source.

## Implementation structure

| Path | Responsibility |
| --- | --- |
| `src/gsim_gap_diagnostic/inputs.py` | Visible target/donor input construction and feature channels. |
| `reconstruction.py` | IDW, Ridge, random forest, Extra Trees and XGBoost. |
| `windows.py`, `replay.py` | Historical-window selection and donor-availability replay. |
| `pipeline.py` | Current reconstruction and within-station historical evidence. |
| `diagnosis.py`, `_history_model.py` | M/H mapping, independent calibration and three-state decisions. |
| `spatial.py`, `evaluation.py` | Spatial folds, station/block weighting and uncertainty evaluation. |
| `controls.py`, `reductions.py` | Input preflight, supervision support and information reduction. |
| `selection.py` | Algorithm selection from visible historical performance. |
| `optional_neural.py` | Supplementary neural reconstruction configurations. |
| `configs/frozen_models/` | 50 transparent M JSON models for the two requirements. |
| `examples/` | Real cases, synthetic inputs and expected outputs. |
| `scripts/` | Full-result replay and reconstruction entry points. |
| `provenance/` | Original-function sources and numerical reproduction checks. |

Paths after the first row in the table refer to the same source package.
See [method details](docs/method.md) for assumptions, defaults, and interpretation.
The source manifest identifies the original functions retained in this portable
implementation. Plotting and manuscript-production code are outside this repository.

## Verified release

Release checks compare the portable reconstruction with the original functions,
the frozen JSON maps with the archived probabilities/decisions, and the two real
examples with their historical-window references. Numerical verification reports
are included under `provenance/` and `examples/real_cases/reproduction_audit.json`.

## Citation and licenses

Software: [MIT](LICENSE). GSIM-derived example data and the accompanying data
release: [CC BY-SA 3.0](LICENSE-DATA.txt), with the original PANGAEA datasets cited.
Use [CITATION.cff](CITATION.cff) for this software. The accompanying dataset
metadata retains the seven manuscript authors and links the upstream datasets.
