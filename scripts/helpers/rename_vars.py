import os
from tempfile import NamedTemporaryFile

import xarray as xr

merged_dir = "/data2/ntsolis/merged"

rename_map = {
    "VHM0_unc": "VHM0_target",
    "VHM0_gt": "VHM0_without",
    "VTM02_unc": "VTM02_target",
    "VTM02_gt": "VTM02_without",
}

for fname in sorted(os.listdir(merged_dir)):
    if not fname.endswith(".nc"):
        continue

    fpath = os.path.join(merged_dir, fname)
    ds = xr.open_dataset(fpath, chunks="auto")

    present = {old: new for old, new in rename_map.items() if old in ds}

    if not present:
        print(f"⏭  {fname:30s} — nothing to rename")
        ds.close()
        continue

    fixed = ds.rename(present)  # <— the single line that does the work
    ds.close()  # close original to release file handle

    # write atomically: first to a temp file, then move over the original
    with NamedTemporaryFile("wb", dir=merged_dir, delete=False) as tmp:
        temp_name = tmp.name
    fixed.to_netcdf(temp_name, mode="w")
    os.replace(temp_name, fpath)
    fixed.close()

    print(f"✓  {fname:30s}  →  renamed {list(present.keys())}")
