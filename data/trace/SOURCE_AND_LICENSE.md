# Source and license

The included trace segment is derived from:

Fernandez-Montes, A., and Fernandez Cerero, D. (2024). *DataCenter-Traces-Datasets* (Version v2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.14564935

The Zenodo record states that the Alibaba 2018 machine-usage files were processed from the Alibaba Cluster Trace 2018 source and are licensed under the Creative Commons Attribution 4.0 International license.

Packaged transformation:

1. Download `machine_usage_days_1_to_8_grouped_300_seconds.csv`.
2. Retain contiguous source rows 1040-1339 without changing source utilization values.
3. Add `bin_index` and `source_row_index` columns.
4. Use `cpu_util_percent` as the replay adapter input.

License text: https://creativecommons.org/licenses/by/4.0/
