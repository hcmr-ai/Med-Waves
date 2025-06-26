import os

import xarray as xr

dir_A = "/data2/ocean2/REANALYSIS/VAL"
dir_B = "/data2/ocean2/RAN_ML/VAL"
outdir = "/data2/ntsolis/merged"
os.makedirs(outdir, exist_ok=True)

# variables we want to *keep* from either source
base_vars = ["WSPD", "WDIR", "VMDR"]  # identical
pair_vars = ["VHM0", "VTM02"]  # differ between dirs

suffix_map = {dir_A: "_unc", dir_B: "_gt"}


def build_path(basedir, fname):
    return os.path.join(basedir, fname)


for fname in sorted(os.listdir(dir_A)):
    if not (fname.startswith("WAVEAN2022") and fname.endswith(".nc")):
        continue

    file_A = build_path(dir_A, fname)
    file_B = build_path(dir_B, fname)

    ds_A = xr.open_dataset(file_A, chunks="auto", engine="netcdf4")
    ds_B = xr.open_dataset(file_B, chunks="auto", engine="netcdf4")

    # rename pair vars
    rename_A = {v: f"{v}{suffix_map[dir_A]}" for v in pair_vars}
    rename_B = {v: f"{v}{suffix_map[dir_B]}" for v in pair_vars}

    # keep base vars + renamed pair vars
    keep_A = base_vars + pair_vars
    keep_B = pair_vars

    ds_A_sel = ds_A[keep_A].rename(rename_A)
    ds_B_sel = ds_B[keep_B].rename(rename_B)

    # align
    ds_A_al, ds_B_al = xr.align(ds_A_sel, ds_B_sel, join="exact")

    merged = xr.merge([ds_A_al, ds_B_al])

    out_path = os.path.join(outdir, fname)
    merged.to_netcdf(out_path)

    print(f"✔︎ {fname}  →  {out_path}")
