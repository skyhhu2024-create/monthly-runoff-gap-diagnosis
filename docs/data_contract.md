# Input and output data contract

All example paths below are relative to the repository root. CSV files use
UTF-8, one header row, decimal points, and blank cells for missing numbers.
JSON uses `null` for missing values, never non-standard `NaN` tokens.

## Monthly reconstruction inputs

`monthly_flow.csv` is a wide monthly table:

```csv
date,target,donor_1,donor_2
1995-01,23.1,18.9,28.0
1995-02,,19.5,29.2
```

| Field | Meaning |
| --- | --- |
| `date` | Unique, ascending `YYYY-MM` calendar month. Include every month; do not compress missing periods. |
| `target` | Visible target discharge. Every month in the requested gap must be missing. |
| Other flow columns | Candidate donor series, aligned by date. Missing donor observations remain blank. Up to 100 candidates. |

All series must use consistent flow units. Finite numbers and missing values
are supported; infinite values are invalid. The public synthetic example uses
arbitrary synthetic units and does not reproduce a GSIM station.

`station_metadata.csv` identifies the same columns:

```csv
station_id,latitude,longitude
target,45.0,10.0
donor_1,45.03,10.04
donor_2,45.06,10.08
```

Coordinates are geographic degrees. The `station_id` must exactly match its
flow-column name. Optional additional metadata do not become predictors. Keep
physical donor identities stable across the current gap and historical replay.
The target is the `target` column in this interface; a GSIM identifier can be
retained separately in a case registry.

CLI gap endpoints are **inclusive calendar months**. Internal array windows
are **zero-based start-inclusive, end-exclusive indices**. For example,
`2004-01` through `2005-12` has length 24 and corresponds to `[108,132)` in
a calendar beginning at `1995-01`. The case's leading/internal/trailing position
is set explicitly to preserve its observation-support direction.

## Case registry

For a study with many cases, retain a separate registry with a stable `case_id`,
`station_id`, gap dates/indices, gap length, position, candidate donor identities,
spatial `block`, source population, and outer-fold assignment. The paper's native
and artificial gaps are distinct registered populations. Each case–algorithm–seed
combination is a distinct reconstruction outcome; this is the join key for its
features, predictions, labels and diagnostic result.

The exact input masks, candidate order, seed and software environment affect
reproducibility. Preserve the case identifier because stable hashes also resolve
window-order and ranking ties. Do not use coordinates rounded for display as
reconstruction inputs if exact archived coordinates are available.

## Historical feature table

The portable feature names follow the original analysis:

| Column | Meaning |
| --- | --- |
| `case_id`, `station_id`, `algorithm`, `seed` | Stable identification. |
| `block`, `split` | Spatial block and role; portable fitting uses `train` and `calibration`. |
| `diagnostic_available` | Visible-data qualification recorded by preprocessing. |
| `hist_window_count` | Number of selected historical windows. |
| `hist_valid_count` | Number with evaluable R/NSE outcomes. |
| `hist_NSE_median`, `hist_R_median` | Medians among evaluable trials. |
| `hist_NSE_iqr`, `hist_R_iqr` | Interquartile ranges among evaluable trials. |
| `hist_joint_rate` | Historical fraction satisfying the study success event. |

M needs a valid count of at least one, a finite median historical NSE, and the
visible-data qualification. H consumes the six summary features listed in
`docs/method.md`. Feature extraction must finish before current-gap labels are
joined. Historical trial truth is allowed only for that trial's score; target
truth inside the current query must never enter feature extraction.

## Labels and outcome evaluation

Artificial-gap labels are kept separately and joined with `case_id`, `algorithm`
and `seed`. A known success label is defined by the selected use requirement;
an unknown outcome remains unknown. The portable fitting label table requires
`R`, `NSE`, `pass_R`, `pass_joint`, `evaluable`, and `model_available` in
addition to the join keys. The paper's A success label is `pass_joint`. B labels
add the positive-skill condition against visible monthly climatology.

The synthetic file `validation_truth.csv` is separated for precisely this
reason. It may be read after reconstruction/evidence/decisions have been stored
to demonstrate evaluation; it is not a reconstruction input. Native missing
intervals have no corresponding reference observations.

## Model JSON

An M model is a transparent JSON object with `format_version`, `method`,
`algorithm`, `seed`, `event`, `score_column`, an `isotonic` object containing
`x_thresholds` and `y_thresholds`, and a `policy` containing `threshold`.
The arrays must have equal lengths, ascending score knots and nondecreasing
probabilities between zero and one. Fitting metadata can record TRAIN/CAL
roles and population counts. An endpoint probability of one is the fitted
mapping output, not a declaration of deterministic success.

The supplied fold-1 RF model serves the four real worked examples. Deploying
to a different population requires suitable TRAIN/CAL evidence for the selected
algorithm, seed policy and success event.

## Release examples and the data archive

`examples/real_cases` contains self-contained monthly `MEAN` input subsets for
the two manuscript cases, original 100-candidate donor pools, metadata, derived
window scores, diagnostic features, model knots and expected decisions. These
monthly inputs come from the public GSIM time-series indices dataset and retain
its CC BY-SA 3.0 licence, with upstream dataset attribution and extraction
changes recorded in `examples/real_cases/ATTRIBUTION.md`. Two additional cases
show the other diagnostic states using derived evidence. A separately labelled
synthetic example is also self-contained. The Zenodo release contains the full
study inputs, result tables and provenance. Original provider daily discharge
records are not included; software licensing does not grant rights to them.
