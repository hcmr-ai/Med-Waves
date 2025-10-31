import os, torch, glob
from tqdm import tqdm

PT_DIR = "/opt/dlami/nvme/preprocessed"
bad_files = []

pt_files = sorted(glob.glob(os.path.join(PT_DIR, "WAVEAN2021*.pt")))
print(f"🔍 Checking {len(pt_files)} .pt files...")

for f in tqdm(pt_files, desc="Validating .pt files"):
    try:
        _ = torch.load(f, map_location="cpu")
    except Exception as e:
        print(f"\n❌ Corrupt: {f} ({e})")
        bad_files.append(f)

print(f"\n✅ Done. Found {len(bad_files)} corrupt files.")
if bad_files:
    with open("corrupt_pt_files.txt", "w") as logf:
        logf.write("\n".join(bad_files))
    print("📝 Saved list to corrupt_pt_files.txt")
