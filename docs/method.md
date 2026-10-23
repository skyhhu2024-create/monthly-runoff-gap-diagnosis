# Method and decisions

This software implements the gap-level diagnosis described in *Diagnosing the
recoverability of monthly streamflow gaps using within-station pseudo-gaps*.
The operational question is whether the information currently available for a
specified missing interval supports a reconstruction meeting a chosen use
requirement. The diagnostic evidence comes from historical reconstructions at
the same target station.

## 1. Define the visible problem

Provide the target's monthly observations, candidate donor observations and
coordinates, a continuous calendar, and the current gap. The target is missing
throughout that interval. Other genuine missing observations stay missing. The
paper uses 1995–2015, artificial lengths of 12, 24, or 36 months, and the nearest
100 geographically located candidate donors from the registered anchor pool
(at least 90% observed coverage; evaluated target stations excluded). The
portable interface accepts an already screened candidate pool. Coordinates
support candidate selection and inverse-distance
weighting; rainfall, catchment attributes, and other external predictors are
not reconstruction inputs.

The target's observed months provide training responses, pairwise donor
calibration and a residual boundary correction. Target discharge is not a
predictor channel. A leading gap uses later target support, an internal gap
uses both sides, and a trailing gap uses earlier support.

## 2. Select historical pseudo-gaps

Find fully observed target intervals of the current gap's length that do not
overlap the current gap. Temporarily hide both the current and historical
intervals, preserve the appropriate support direction, and retain at least
12 observed target months.

Candidate windows are ranked by the equally weighted mean of four distances:

1. Circular starting-month distance, divided by six.
2. Relative difference in retained target-month counts.
3. Relative difference in mean target–donor concurrent-month counts.
4. Absolute difference in donor coverage during the query.

For counts, the relative difference is `abs(a-b) / max(a,b,1)`. Select at most
four non-overlapping windows in that order. Stable case-ID hashes resolve
ties. Window selection uses visible masks, calendar position and support
counts, without using the hidden current outcome.

For each historical trial, replay the current gap's donor availability and
the preceding 11 months by physical donor identity, intersecting this with
the historical availability. Reselect donors and refit preprocessing and the
reconstruction method after replay. Historical observations are retained
separately only for evaluating that historical trial.

## 3. Reconstruct each trial

Select up to eight positively correlated donors with at least six visible
concurrent target–donor months. All five main algorithms use this visible
candidate information:

| Algorithm | Implementation |
| --- | --- |
| IDW | Pairwise rescaling to the target's concurrent mean and standard deviation, followed by inverse squared-distance weighting of actually observed donors. |
| Ridge | Ridge penalty 10; intercept is unpenalised. |
| RF | 256 trees, minimum leaf size 3, all predictors considered per split. |
| ET | 256 extremely randomised trees, minimum leaf size 3, all predictors considered per split. |
| XGB | 300 trees, depth 3, learning rate 0.03, row and column sampling 0.8, L2 penalty 1. |

The four learners receive 28 monthly channels: four annual/semiannual harmonic
terms and three channels for each of eight donors (standardised discharge,
deseasonalised anomaly, missingness indicator). The current month and preceding
11 months produce 336 flattened predictors. Pairwise scales and seasonal curves
are fitted on visible overlap. Missing donor values receive seasonal estimates,
their anomaly is zero, and their missingness flag remains one. The visible
fit residuals provide a lag-one boundary correction. Predictions are clipped
at zero. IDW does not impute missing donor observations: if a query month has
no observed active donor, that month's IDW prediction remains unavailable.

The supplementary LSTM and TCN use the same 12-by-28 sequences. They are
station-level configurations in the paper, not globally pretrained networks.
Their role and exact implementations are retained separately from the five
main methods.

## 4. Form evidence and define the event

Each evaluable historical trial yields a correlation and an NSE. The main
diagnosis **M** uses the median historical NSE as its score. The comparison
diagnosis **H** uses six features: valid trial count, median R, median NSE,
historical success fraction, and the interquartile ranges of R and NSE.

Requirement A is a whole-gap NSE of at least 0.5. For a finite non-constant
paired sample, the exact identity

`NSE = R² − (alpha − R)² − delta²`

relates NSE to correlation, the predicted-to-observed standard-deviation ratio
`alpha`, and standardised mean bias `delta`. Positive NSE requires positive
correlation, so A also meets the original `R >= 0.7` requirement. Requirement
B additionally requires lower squared error than monthly climatology estimated
from visible target observations. Each event has its own fitted probability
mapping and policy.

Evidence is unavailable if there is no eligible historical window or no
evaluable selected trial. This is an evidence state, not a probability of zero.
The sparse synthetic example and the native AU_0001379 example illustrate it.

## 5. Learn probabilities and an acceptance policy

For M, fit increasing isotonic regression from historical NSE to the registered
artificial-gap success label. Use TRAIN records only. The stored JSON model
contains the isotonic score/probability knots. Prediction linearly interpolates
between knots and clips to endpoint probabilities outside their score range,
matching scikit-learn's fitted isotonic predictor.

Choose the acceptance threshold using CAL only. Among thresholds meeting
85% precision and 80% success recall, select greatest coverage. If none meets
both targets, maximise the smaller attained-to-target ratio and retain an
explicit unmet-target flag. H uses spline logistic regression followed by CAL
Platt calibration. TEST outcomes are used for evaluation after decisions have
been fixed.

The paper holds out five groups of 5-degree spatial blocks. For outer TEST
fold `k`, CAL is `(k + 1) % 5`, and the other three folds are TRAIN. Mapping
and threshold fitting use only the original 3262-station cohort. All 8152
registered stations contribute once in their TEST role. The fold-1 example
model therefore uses TRAIN folds 0, 3, and 4 and CAL fold 2.

## 6. Return a gap-level result

| State | Rule | Operational reading |
| --- | --- | --- |
| `accept` | Evaluable historical evidence and calibrated probability at or above the frozen threshold. | The available evidence supports accepting this reconstruction under the selected event. |
| `low_probability_reject` | Evaluable evidence and probability below threshold. | Current information does not support acceptance under this policy. |
| `evidence_insufficient` | No eligible/evaluable historical evidence. | The diagnostic system abstains. |

The original result archive's equivalent `state` names are `adopt`,
`reject_low_probability`, and `evidence_insufficient`.

The optional algorithm-selection policy chooses the largest valid median
historical NSE across IDW, Ridge, RF, ET, and XGB, breaking ties in that order.
It receives a separately fitted mapping and threshold for its own selected
outcome. If no algorithm has evidence, it retains the prespecified RF
reconstruction and abstains diagnostically.

## Evaluation and scope

Acceptance precision is confirmed accepted successes divided by all accepted
gaps. Success recall is accepted confirmed successes divided by all confirmed
successes, including successes without diagnostic evidence. Coverage is accepted
gaps divided by all registered gaps. Unknown outcomes do not become failures
or successes; accepted unknowns remain in the precision denominator. Spatial
block weighting and paired whole-block resampling address the uneven evaluation
network.

The real-gap examples are applications without gap-internal target truth.
Their probabilities and decisions are outputs of the fitted diagnostic system.
Full study populations, controls, spatial strata and result tables belong to
the companion data archive. The executable synthetic input exercises the code;
its performance is not an empirical result of the paper.
