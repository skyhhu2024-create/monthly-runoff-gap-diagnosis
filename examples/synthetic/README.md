# Self-contained synthetic input

Every flow value in this directory is generated from the script
`generate_inputs.py`. It is unrelated to any GSIM observation. This example
exercises the visible-data reconstruction and historical-evidence pipeline;
it is not a result from the study.

`input/monthly_flow.csv` contains 252 consecutive months, eight donors, and an
internal 24-month target gap from January 2004 through December 2005.
`monthly_flow_sparse.csv` retains only every third outside-gap target month.
Both use the same donor observations and synthetic target-generating process.

After installing the package, run from the repository root:

```shell
gsim-gap reconstruct --flow examples/synthetic/input/monthly_flow.csv --metadata examples/synthetic/input/station_metadata.csv --gap-start 2004-01 --gap-end 2005-12 --case-id synthetic-dense --station-id synthetic-target --algorithm rf --output outputs/synthetic-dense
gsim-gap reconstruct --flow examples/synthetic/input/monthly_flow_sparse.csv --metadata examples/synthetic/input/station_metadata.csv --gap-start 2004-01 --gap-end 2005-12 --case-id synthetic-sparse --station-id synthetic-target --algorithm rf --output outputs/synthetic-sparse
```

Each output directory must be new. `python -m gsim_gap_diagnostic` is equivalent
to the installed `gsim-gap` command. The dense example produces four valid
historical windows; the sparse variant produces none because it has no complete
24-month observed historical interval. The sparse current reconstruction can
still be generated; a reconstruction and evidence supporting its acceptance
are different outputs.

No probability model is attached to these artificial inputs. The pipeline
therefore saves reconstruction and historical evidence without inventing a
calibrated probability. The fully calibrated real-case examples are in
`../real_cases`.

The distributed `expected_output` directories record the tested primary-seed
configuration. Dense median historical NSE is approximately `0.9886324488580072`;
the sparse median is missing, with zero selected and zero valid trials.
`validation_truth.csv` is separate and must never be supplied as a feature or
reconstruction input. It supports only a subsequent software-evaluation example.

To regenerate the inputs with the fixed seed:

```shell
python examples/synthetic/generate_inputs.py
```

The generator code uses the repository's MIT licence. The distributed synthetic
CSV files and expected-result data use CC BY-SA 3.0, as do the release's other
example data. Their provenance is explicitly synthetic.
