# GSIM attribution and reuse

The monthly discharge inputs and coordinates in `BR_0001891/` and
`US_0002700/` are selected from the following public GSIM datasets:

- Gudmundsson, Lukas; Do, Hong Xuan; Leonard, Michael; Westra, Seth (2018):
  *The Global Streamflow Indices and Metadata Archive (GSIM) - Part 2: Time
  Series Indices and Homogeneity Assessment* [dataset]. PANGAEA.
  [https://doi.org/10.1594/PANGAEA.887470](https://doi.org/10.1594/PANGAEA.887470).
- Do, Hong Xuan; Gudmundsson, Lukas; Leonard, Michael; Westra, Seth (2018):
  *The Global Streamflow Indices and Metadata Archive - Part 1: Station catalog
  and Catchment boundary* [dataset]. PANGAEA.
  [https://doi.org/10.1594/PANGAEA.887477](https://doi.org/10.1594/PANGAEA.887477).

Both datasets are licensed under **Creative Commons Attribution-ShareAlike
3.0 Unported (CC BY-SA 3.0)**. The monthly inputs, coordinates, masks and derived
case outputs supplied here retain that licence:
[licence summary](https://creativecommons.org/licenses/by-sa/3.0/) and
[legal code](https://creativecommons.org/licenses/by-sa/3.0/legalcode).
The repository's MIT software licence applies to its code, not to these data.

These are **monthly `MEAN` indices**, rather than the original daily discharge
records supplied to GSIM by providers. Each case includes the target and its
original 100-candidate donor pool. This preserves the original mask-based
window matching and donor reselection; reducing the file to only the final
eight donors can change those operations.

Changes made for this release:

1. Select the `MEAN` column and the January 1995–December 2015 calendar.
2. Keep the last row for a duplicated calendar month, matching the study's
   registered parsing rule; normalise nonfinite values to missing.
3. Mask target values in the query interval. For the artificial BR_0001891
   case, put the withheld values in a separate `validation_truth.csv` file.
4. Give columns portable aliases (`target`, `donor_001`–`donor_100`) and
   retain their GSIM identities in `station_metadata.csv`.
5. Export reproducible expected historical scores, features, reconstructions
   and diagnostic decisions, without adding provider daily records.

Every included monthly source was checked against the study calendar before
export. The per-case `manifest.json` records the original `.mon` filename and
SHA-256 checksum for each of the 101 series. `reproduction_audit.json` records
the executable checks and software versions. GSIM's upstream data providers
and archive authors retain their attribution as documented by the source
archive and station catalogue.
