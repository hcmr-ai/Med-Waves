---
name: "medwav-blobfuse-mounts"
description: "Use when working in the Med-WAV repo and the task is to install blobfuse2, create or repair Azure Blob Storage mounts, or explain how to use `scripts/azure/setup_blobfuse_mounts.sh`. This skill covers the Med-WAV default storage account and containers, the supported environment overrides, the required sudo/escalation boundary, and the verification steps for `/mnt/blobstorage` and `/mnt/blobstorage-scalers`."
---

# Med-WAV Blobfuse Mounts

Use this skill when the user wants to mount the Med-WAV Azure Blob containers with blobfuse2, troubleshoot those mounts, or adapt the mount script for a different storage account, container, or mount path.

Primary script:
- `scripts/azure/setup_blobfuse_mounts.sh`

## What the script does

The script provisions two blobfuse2 mounts with Med-WAV defaults:
- storage account: `medwavdatastorageneu`
- main container: `medwav-data` mounted at `/mnt/blobstorage`
- scalers container: `scalers` mounted at `/mnt/blobstorage-scalers`

It will:
- install `blobfuse2` and `fuse3` if missing
- prompt for `AZURE_STORAGE_KEY` if not already set
- write per-container blobfuse2 config files in `$HOME`
- create cache directories under `/mnt/blobfuse2-cache`
- mount both containers
- verify each mount with `mountpoint`

## Required operating assumptions

- The installer path only supports Ubuntu because it uses `/etc/os-release` and the Microsoft Ubuntu package feed.
- Running the script normally requires `sudo`.
- Mount creation, package installation, and writes under `/mnt` are outside the workspace sandbox. If you are operating through Codex, request escalation before running it.
- Treat the storage key as a secret. Prefer passing it via environment variable or letting the script prompt silently. Do not print or log the key.

## Default workflow

1. Inspect `scripts/azure/setup_blobfuse_mounts.sh` before changing behavior.
2. Confirm whether the user wants the default Med-WAV mounts or custom overrides.
3. If the task is to execute the script, explain that it needs elevated privileges and likely writes outside the repo.
4. Use the stock command unless the user asked for overrides:

```bash
AZURE_STORAGE_KEY='…' ./scripts/azure/setup_blobfuse_mounts.sh
```

5. If the user does not want to expose the key in the shell history, omit the variable and let the script prompt:

```bash
./scripts/azure/setup_blobfuse_mounts.sh
```

6. After execution, verify:
- `mountpoint -q /mnt/blobstorage`
- `mountpoint -q /mnt/blobstorage-scalers`
- optional sanity check: `ls /mnt/blobstorage` and `ls /mnt/blobstorage-scalers`

## Supported overrides

The script is parameterized with environment variables. Use these rather than patching the script when the user only needs a different target:

- `STORAGE_ACCOUNT`
- `CONTAINER_MAIN`
- `CONTAINER_SCALERS`
- `MOUNT_MAIN`
- `MOUNT_SCALERS`
- `CACHE_ROOT`
- `CONFIG_MAIN`
- `CONFIG_SCALERS`
- `AZURE_STORAGE_KEY`

Example:

```bash
STORAGE_ACCOUNT=myaccount \
CONTAINER_MAIN=my-data \
CONTAINER_SCALERS=my-scalers \
MOUNT_MAIN=/mnt/my-data \
MOUNT_SCALERS=/mnt/my-scalers \
AZURE_STORAGE_KEY='…' \
./scripts/azure/setup_blobfuse_mounts.sh
```

## Behavior details worth remembering

- If `blobfuse2` already exists, the install step is skipped.
- If a mount path is already mounted, the script prints `Already mounted` and does not remount it.
- The script writes config files containing the account key and then locks them down with `chmod 600`.
- The cache root is shared, but each container gets its own cache subdirectory.

## Troubleshooting checklist

- If the installer fails early, check the machine is Ubuntu and has `sudo`.
- If package install fails, check network access to `packages.microsoft.com`.
- If mount commands fail, verify the storage key and container names.
- If permissions look wrong under `/mnt`, check the `sudo chown -R "$USER:$USER"` step and whether the mount points already existed with conflicting ownership.
- If the mount exists but access is empty or inconsistent, inspect the generated blobfuse2 YAML files in `$HOME`.

## When to edit the script

Edit `scripts/azure/setup_blobfuse_mounts.sh` only when the workflow itself must change, for example:
- adding another default container mount
- changing install behavior
- changing verification logic
- changing default config layout

For one-off execution differences, prefer environment variable overrides instead of editing the script.
