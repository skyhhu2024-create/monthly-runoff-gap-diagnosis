# Implementation architecture

The package separates reconstruction evidence from current-gap outcome labels.
The same visible-input path serves the registered artificial gaps and native
missing intervals. Figure-generation and manuscript-formatting code are outside
the public method implementation.

```text
monthly observations + coordinates + registered gap
                       |
                  visible inputs
                       |
           select historical pseudo-gaps
                       |
       replay masks / reselect donors / refit
                       |
      current predictions + historical trial scores
                       |
             visible historical features
                       |
               frozen M or H model
                       |
       probability + evidence state + decision

separate artificial-gap reference observations
                       |
        labels joined only for TRAIN/CAL fitting
            or saved-decision evaluation
```

## Modules

| Module | Responsibility |
| --- | --- |
| `cli.py` | Read portable CSV inputs; expose reconstruction, fitting, diagnosis, spatial cross-validation, and evaluation commands. |
| `inputs.py` | Visible donor selection, pairwise calibration, harmonic and missingness channels, 12-month sequences, and residual boundary correction. |
| `windows.py` | Historical-window eligibility, four-component matching, non-overlap and deterministic ordering. |
| `replay.py` | Replay current donor masks into the historical query and preceding context, then rebuild inputs. |
| `reconstruction.py` | IDW, Ridge, RF, ET and XGB reconstruction from shared prepared inputs. |
| `optional_neural.py` | Supplementary station-level LSTM and TCN reconstruction configurations. |
| `pipeline.py` | Execute one current gap and its historical trials; return evidence features and predictions. |
| `scores.py` | Paired R/NSE, visible monthly-climatology skill, success labels and historical summaries. |
| `diagnosis.py` | M isotonic fitting, H fitting, independent CAL policy selection, and frozen-model prediction. |
| `_history_model.py` | Original H feature transforms, spline logistic model and probability calibration. |
| `spatial.py` | Spatial block identity and role assignment helpers. |
| `evaluation.py` | Cross-validation, precision, recall, coverage and block-resampling evaluation. |
| `selection.py` | Historical-score algorithm selection with fixed tie order. |
| `controls.py` | IDW historical input precheck and learner supervision control. |
| `reductions.py` | Reproduce the three visible-information reduction types. |
| `_io.py` | Safe JSON/CSV representation, input fingerprints and new output directories. |

## Portable commands

Install the package from the repository root:

```shell
python -m pip install -e .
```

The RF real-case demonstration then runs the complete visible-input path:

```shell
python examples/real_cases/reproduce.py --output outputs/real_cases
```

The same interface accepts new, appropriately prepared data:

```shell
gsim-gap reconstruct --flow my_monthly_flow.csv --metadata my_stations.csv --gap-start 2004-01 --gap-end 2005-12 --case-id my-gap --station-id my-target --algorithm rf --output outputs/my-gap
```

This saves the current reconstruction, selected historical windows, historical
trial scores, visible summary features and an input-provenance receipt. Add
`--model my_model.json` to apply an already fitted M model for the same
algorithm, primary seed and success event. Without a model, reconstruction and
historical evidence are saved without a calibrated acceptance decision.

To fit a model from explicitly separate spatial TRAIN and CAL populations:

```shell
gsim-gap fit --features my_train_cal_features.csv --labels my_train_cal_labels.csv --method M --event original_event --output outputs/my-model
gsim-gap diagnose --features my_application_features.csv --model outputs/my-model/model.json --output outputs/my-decisions.csv
```

`fit` handles one algorithm and seed at a time. TRAIN and CAL stations and
spatial blocks must be disjoint. `diagnose` accepts visible historical features
and rejects current outcome or prior probability columns. See the complete
field contract in `data_contract.md`.

For the paper's five-fold spatial experiment, `cross-validate` consumes the full
registered feature table, labels and case registry; each outer fold gets its
own TRAIN mapping and CAL threshold. `evaluate` consumes already persisted
decisions and separately held outcome labels. The data archive preserves the
registered populations and result tables needed to check those experiments.

The optional XGBoost dependency is installed with `pip install -e ".[xgboost]"`.
The supplementary neural implementation is available through the Python module
with the optional `neural` dependency. The main CLI algorithms correspond to
the five methods used in the main-text robustness comparison.

## Reproduction levels

1. **Derived decision check:** `examples/real_cases/verify_expected.py` uses
   only the Python standard library to reproduce all three states from released
   historical scores and model knots.
2. **Complete case reconstruction:** `examples/real_cases/reproduce.py` rebuilds
   the two real examples from public monthly GSIM inputs and compares historical
   evidence, current predictions, probabilities and decisions.
3. **Study reanalysis:** use the full result/input archive, registered case
   masks and fold assignments. Refit only within the designated TRAIN/CAL roles,
   and preserve the selected event, cohort, candidate pools and seed policy.

These levels answer different reproduction questions. The synthetic example
provides an additional inspectable input generator, with its provenance clearly
distinguished from the real study cases.
