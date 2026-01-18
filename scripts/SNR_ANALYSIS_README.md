# SNR Analysis for Wave Model Evaluation

## Overview

The `calculate_snr.py` script computes Signal-to-Noise Ratio (SNR) per wave height bin and analyzes the correlation between model error and SNR. This helps determine if SNR could be useful as an input feature for your TransUNet model.

## What It Does

1. **Loads real data** from parquet files (local or S3)
2. **Computes SNR per wave bin** - Signal variance / Noise variance
3. **Analyzes correlations** - How does error magnitude relate to SNR?
4. **Generates 4 plots**:
   - Plot 1: Mean SNR (dB) per wave height bin
   - Plot 2: RMSE vs SNR scatter plot
   - Plot 3: Per-bin error-SNR correlation coefficients
   - Plot 4: Pixel-level error vs SNR hexbin density plot
5. **Prints detailed table** with all metrics

## Key Interpretation

- **If overall correlation > 0.6**: Strong relationship → SNR as feature could help TransUNet!
- **If per-bin correlation high in extremes**: Local SNR features may be needed
- **High SNR (>10 dB)**: Good signal quality, model should perform well
- **Low SNR (<5 dB)**: Poor signal quality, high errors may be unavoidable

## Usage Examples

### Basic Usage (Analyze 2021 from S3)
```bash
python scripts/calculate_snr.py --year 2021
```

### Analyze Specific Year with File Limit
```bash
# Analyze first 50 files from 2022
python scripts/calculate_snr.py --year 2022 --max-files 50
```

### Use Custom Columns
```bash
# If your ground truth is in 'corrected_VHM0' instead of 'vhm0_y'
python scripts/calculate_snr.py --year 2021 \
    --ground-truth-col corrected_VHM0 \
    --reference-col VHM0
```

### Analyze from Local Directory
```bash
python scripts/calculate_snr.py \
    --input-dir /path/to/data/hourly/ \
    --year 2021
```

### Custom Output Path
```bash
python scripts/calculate_snr.py --year 2021 \
    --output-path results/snr_analysis_2021.png
```

### Full Year Analysis (All Files)
```bash
# This will process all ~365 files from 2021
python scripts/calculate_snr.py \
    --input-dir s3://medwav-dev-data/parquet/hourly/ \
    --year 2021
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `s3://medwav-dev-data/parquet/hourly/` | Directory with parquet files |
| `--year` | `2021` | Year to analyze |
| `--ground-truth-col` | `vhm0_y` | Column name for ground truth |
| `--reference-col` | `VHM0` | Column name for reference/uncorrected |
| `--max-files` | `None` | Max files to process (None = all) |
| `--output-path` | `snr_analysis_complete.png` | Output plot path |
| `--min-samples` | `50` | Min samples per bin |

## Output

### Console Output
```
================================================================================
LOADING DATA FROM PARQUET FILES
================================================================================
Input directory: s3://medwav-dev-data/parquet/hourly/year=2021/
Ground truth column: vhm0_y
Reference column: VHM0
File pattern: WAVEAN2021*.parquet
Found 365 files to process

Loading parquet files: 100%|████████████████| 365/365 [05:23<00:00, 1.13it/s]

✓ Data loaded successfully!
  Total samples: 45,678,910
  Ground truth range: [0.012, 14.567]
  Reference range: [0.001, 14.892]

... (SNR analysis results) ...

========================================================================================================================
DETAILED SNR ANALYSIS TABLE
========================================================================================================================
Bin          N_samples    Signal_Var   Noise_Var    SNR_dB     RMSE       MAE        Bias       Bin_Corr    
------------------------------------------------------------------------------------------------------------------------
0.0-1.0m     12,345,678   0.0842       0.0123       8.35       0.1234     0.0987     -0.0123    0.234       
1.0-2.0m     15,234,567   0.2341       0.0234       10.01      0.1567     0.1234     -0.0234    0.345       
...

OVERALL CORRELATION (Error-SNR): 0.7234 (p=0.00e+00, n=45,678,910)
→ STRONG RELATIONSHIP: SNR as feature could significantly help TransUNet!
========================================================================================================================
```

### Plot Output
A single PNG file with 4 subplots showing comprehensive SNR analysis.

## Integration with evaluate_bunet.py

If you want to use this during model evaluation, you can integrate it by:

1. In `ModelEvaluator.evaluate()`, save arrays:
```python
# Save arrays for SNR analysis
np.save(self.output_dir / 'ground_truth.npy', 
        np.array(self.plot_samples["y_true"]))
np.save(self.output_dir / 'reference.npy', 
        np.array(self.plot_samples["y_uncorrected"]))
```

2. Then run SNR analysis separately on the saved arrays.

## Performance Notes

- **Memory efficient**: Loads only 2 columns from parquet files
- **Progress tracking**: Shows progress bar with tqdm
- **Handles large datasets**: Tested with 45M+ samples
- **S3 streaming**: Reads directly from S3 without downloading
- **Subsampling**: Use `--max-files` to test on subset

## Typical Runtime

- **10 files**: ~15 seconds
- **50 files**: ~1.5 minutes  
- **365 files (full year)**: ~5-8 minutes

## Next Steps After Analysis

1. Review the 4-panel plot
2. Check overall correlation in the table
3. **If correlation > 0.6**: Consider adding SNR as input feature
4. **If per-bin correlations vary**: Consider local/adaptive SNR features
5. Focus model improvements on bins with low SNR

## Troubleshooting

### "No files found" error
- Check your `--input-dir` path
- Verify the `--year` parameter
- Ensure S3 credentials are configured (for S3 paths)

### Column not found error
- Check column names in your parquet files
- Use `--ground-truth-col` and `--reference-col` to specify correct columns

### Memory issues
- Use `--max-files 50` to limit file count
- The script is already memory-efficient (loads only 2 columns)
