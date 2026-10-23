# Real cases: historical evidence to a decision

These four cases use real registered GSIM station identifiers and diagnostic
results from the study. The two manuscript cases, BR_0001891 and US_0002700,
include self-contained monthly GSIM `MEAN` input subsets and expected outputs.
The other two cases illustrate decisions from released derived historical
evidence. The RF algorithm, primary seed
`20261014`, requirement A, and held-out fold 1 are common to all four cases.
The study's internal method name `S_trainiso` corresponds to manuscript M.

Run this dependency-free numerical check from the repository root:

```shell
python examples/real_cases/verify_expected.py
```

It checks the historical NSE medians, uses the original isotonic knots with
linear interpolation and endpoint clipping, and reproduces all three decision
states. Expected output is `status: PASS` for four cases.

For the complete real-input reconstruction, historical pseudo-gap and probability
pipeline, install the package and run:

```shell
python examples/real_cases/reproduce.py --output outputs/real_cases
```

This runs BR_0001891 and US_0002700 from their monthly input tables, recomputes
the four historical reconstructions at each station, applies the frozen M
model, and compares outputs against the released reference files. It reads
no withheld current-gap target truth. The output folder must be new.

To apply the probability mapping directly to all four released feature rows:

```shell
gsim-gap diagnose --features examples/real_cases/visible_features.csv --model examples/real_cases/model_m_rf_fold1.json --output outputs/real_case_decisions.csv
```

Portable decision names are `accept`, `low_probability_reject`, and
`evidence_insufficient`. The original-analysis `state` field uses the equivalent
names `adopt`, `reject_low_probability`, and `evidence_insufficient`.

| Case | Gap | Valid trials | Median historical NSE | Probability | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| BR_0001891, artificial | May 1999–April 2001 | 4 | 0.897895 | 0.953757 | Accept |
| US_0002700, native | October 2000–September 2002 | 4 | 0.962099 | 1.000000 | Accept |
| US_0008293, native | October 2003–September 2005 | 4 | 0.122781 | 0.230159 | Reject: low probability |
| AU_0001379, native | July 1997–May 2000 | 0 | Missing | Missing | Insufficient evidence |

The frozen acceptance threshold is `0.4584264749367909`. BR_0001891 is the
manuscript Figure 2 example; its four historical NSE values are
`0.8926176567483102`, `0.9334855480381087`, `0.8781218247135872`, and
`0.9031717832318628`. The median `0.8978947199900864` maps to
`0.953757225433526`, exceeding the threshold. The current artificial-gap truth
is excluded from this calculation. After the decision is saved, the separate
withheld observations give current-gap R `0.9540468581108347` and NSE
`0.8719874718574405` for this RF reconstruction. This evaluation is saved
separately in the example's expected output.

US_0002700 is the Figure 6 natural-gap example. Its probability of one is the
endpoint output of the empirical isotonic mapping. Its gap-internal target
observations do not exist in the input archive, so no current-gap accuracy can
be computed. US_0008293 and AU_0001379 are release-only illustrations of the
other states. Within internal fold-1 native gaps of each requested state,
24-month cases were preferred where available, then the smallest case-ID hash
was used. The insufficient-evidence example is a 35-month interval because no
24-month example in that fold has that state. Its target history contains no
complete eligible window of the required length.

## Files

| File | Content |
| --- | --- |
| `diagnostic_features_and_expected_decisions.csv` | Six historical features, evidence qualification and expected outputs. Expected probability/state columns are verification references, never fitting inputs. |
| `visible_features.csv`, `expected_decisions.csv` | Separated, directly usable input and reference-output tables. |
| `historical_window_scores.csv` | Each selected trial's index bounds, R, NSE and evaluability. |
| `model_m_rf_fold1.json` | Original non-executable score/probability knots, CAL policy and TRAIN/CAL roles. |
| `case_metadata.json` | Gap dates, example-selection rules and truth availability. |
| `provenance.json` | SHA-256 fingerprints of the analysis sources used to extract these results. |
| `BR_0001891/`, `US_0002700/` | 252-month target and 100-candidate donor tables, coordinates and GSIM identity mapping, source verification, and tested expected outputs. |
| `reproduction_audit.json` | Numerical checks of full reconstruction and all three decision states, including package versions. |
| `ATTRIBUTION.md` | GSIM dataset attribution, CC BY-SA 3.0 licence, and changes made for these examples. |

Index origin is January 1995; starts are inclusive and ends exclusive. The
fold-1 model was fitted on original-cohort TRAIN folds 0, 3 and 4, with threshold
selection in CAL fold 2. It is supplied to reproduce these examples. Full
reconstruction of the two manuscript cases is self-contained in this directory.
The public monthly `MEAN` subsets were checked against each original `.mon`
series and are redistributed under the GSIM dataset's CC BY-SA 3.0 licence;
see `ATTRIBUTION.md`. The original provider daily observations are not included.
The additional synthetic input demonstration is in `../synthetic`.
