# Handover Data Notebooks

This folder contains lean notebooks for showing one sample Med-WAV file across the main dataset representations used in handover:

- `01_locate_sample_file.ipynb`: find the same sample date across Azure and Neptune paths
- `02_describe_sample_file.ipynb`: inspect one chosen artifact in detail
- `03_compare_representations.ipynb`: compare the same sample across `.nc`, `.parquet`, and `.pt`

The notebooks are parameterized around one `WAVEANYYYYMMDD` sample date and support both Azure and Neptune path layouts through shared helpers in `sample_data_helpers.py`.
