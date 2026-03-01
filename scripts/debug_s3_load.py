#!/usr/bin/env python3
"""Quick test to verify S3 .pt file loading works"""
import torch
import s3fs

# Test file path
test_file = "s3://medwav-dev-data/preprocessed/WAVEAN20220101.pt"

print("="*60)
print("Testing S3 .pt file loading...")
print("="*60)

# Test 1: Direct S3FS load
print("\n1. Creating S3FileSystem...")
fs = s3fs.S3FileSystem()
print("✓ S3FileSystem created")

print(f"\n2. Checking if file exists: {test_file}")
exists = fs.exists(test_file)
print(f"   File exists: {exists}")

if not exists:
    print("\n❌ ERROR: File does not exist! Check:")
    print("   - Path is correct")
    print("   - Files were preprocessed successfully")
    print("   - IAM role has S3 read permissions")
    exit(1)

print(f"\n3. Opening file...")
with fs.open(test_file, "rb") as f:
    print("   ✓ File opened")
    print(f"\n4. Loading with torch.load()...")
    data = torch.load(f, map_location="cpu")
    print("   ✓ Torch load successful")

print(f"\n5. Checking data structure...")
print(f"   Keys: {data.keys()}")
print(f"   Tensor shape: {data['tensor'].shape}")
print(f"   Feature columns: {len(data['feature_cols'])} columns")
print(f"   First 5 features: {data['feature_cols'][:5]}")

print("\n" + "="*60)
print("✓ All tests passed! S3 loading works correctly.")
print("="*60)
