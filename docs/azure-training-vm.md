# Azure Training VM

This project commonly trains on Azure `Standard_NV12ads_A10_v5`.

## Baseline VM

Current project standard:
- size: `Standard_NV12ads_A10_v5`
- vCPUs: `12`
- memory: `110 GiB`
- temp disk: `360 GiB`
- accelerator slice: `1/3` of an NVIDIA A10 GPU

From current Microsoft Learn documentation, the `NVadsA10_v5` family uses NVIDIA A10 GPUs and AMD EPYC 74F3V CPUs, and `Standard_NV12ads_A10_v5` is a `12 vCPU / 110 GiB` size with a `360 GiB` temp disk. That series ships with a GRID license and uses partial A10 GPU slices. Sources:
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series

## Recommended Provisioning Choices

Recommended defaults for this repo:
- OS: Ubuntu LTS
  - prefer Ubuntu `22.04 LTS` for a conservative setup
  - Ubuntu `24.04 LTS` is supported in current Microsoft docs, but use it only if you have validated the driver path you want
- authentication: SSH key
- secure boot: disabled
- vTPM / Trusted Launch secure boot path: disabled unless you have a reason to support signed-driver flows
- region: choose a region where `NVadsA10_v5` quota is available
- disk: leave enough OS disk headroom for Poetry, local checkouts, and temporary artifacts

Why secure boot and vTPM are called out:
- Microsoft’s Linux N-series GPU driver guidance explicitly warns that Secure Boot and vTPM can block or hang the driver setup flow for this class of VM unless you follow the signed-driver path carefully.

## Provisioning Checklist

Before creating the VM:
- confirm Azure quota for `NVadsA10_v5` in the target region
- prepare an SSH public key
- know whether you want blob-backed checkpoints and diagnostics mounted immediately
- know whether training data will live on mounted blob storage, another data disk, or a pre-staged local dataset path

At creation time:
- choose `Standard_NV12ads_A10_v5`
- pick Ubuntu LTS
- disable secure boot / vTPM for the simplest NVIDIA path
- ensure outbound internet access exists for driver install, Poetry install, and package setup

## GPU Driver Setup

### Preferred path: Azure-supported NVIDIA GPU Driver Extension

For handover, this should be the default recommendation.

Microsoft’s current guidance for Linux N-series VMs is:
- use the `NvidiaGpuDriverLinux` extension when possible
- `NVads A10 v5` supports GRID drivers, not the older NC/ND CUDA-only path
- the A10 vGPU driver is a unified driver for graphics and compute workloads

Official references:
- Azure N-series GPU driver setup for Linux:
  - https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
- NVIDIA GPU Driver Extension for Linux:
  - https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux

Important current notes from Microsoft Learn:
- `NVads A10 v5` only supports GRID `17.x` or higher
- `vGPU18` is now available for `NVadsA10_v5`
- GRID installation has known issues on Azure kernel `6.11`; downgrading to kernel `6.8` is the documented workaround

### Repo fallback path: pinned manual install scripts

The repo contains:
- [`scripts/azure/install_nvidia_part1.sh`](../scripts/azure/install_nvidia_part1.sh)
- [`scripts/azure/install_nvidia_part2.sh`](../scripts/azure/install_nvidia_part2.sh)

These scripts are not general-purpose installers. They are a pinned operational workaround for a specific VM/user/kernel/driver combination:
- they pin kernel `6.8.0-1025-azure`
- they hardcode a specific GRID installer:
  - `NVIDIA-Linux-x86_64-570.195.03-grid-azure.run`
- they assume the Linux username `azureuser`
- they append Poetry setup to that user’s shell profile

Treat them as:
- a recovery path when the official extension path is not sufficient
- a reference for the kernel/driver combination that worked for this repo

Do not treat them as the first-choice long-term provisioning mechanism.

## Manual NVIDIA Install Runbook

Use this runbook when you are following the repo’s pinned manual driver path.

### Preconditions

Before running the scripts, confirm:
- the VM is Ubuntu
- you can SSH into the machine
- the active user has `sudo`
- outbound internet access works
- the repo is available on the machine
- Secure Boot / vTPM are disabled for the simple unsigned-driver path

These scripts assume:
- the Linux username is `azureuser`
- the working kernel target is `6.8.0-1025-azure`
- the NVIDIA installer URL remains valid

### Why the install is split in two

The install is split because the machine must:
1. install and boot into a specific Azure kernel
2. blacklist `nouveau` and rebuild initramfs
3. reboot
4. only then install the pinned NVIDIA GRID driver

Do not run part 2 until the machine has rebooted and `uname -r` reports `6.8.0-1025-azure`.

### Step 1: prepare kernel and reboot

From the repo root:

```bash
bash scripts/azure/install_nvidia_part1.sh
```

This script:
- installs build tools
- installs `linux-image-6.8.0-1025-azure` and matching headers
- updates GRUB to boot that kernel
- blacklists `nouveau`
- updates initramfs
- disables conflicting services
- reboots the VM

### Step 2: reconnect after reboot

SSH back into the machine and verify the kernel:

```bash
uname -r
```

Expected output:

```bash
6.8.0-1025-azure
```

If the kernel is not correct, do not continue to part 2. Fix boot selection first.

### Step 3: install the NVIDIA GRID driver

From the repo root:

```bash
bash scripts/azure/install_nvidia_part2.sh
```

This script:
- verifies the active kernel
- unloads existing NVIDIA modules
- downloads `NVIDIA-Linux-x86_64-570.195.03-grid-azure.run`
- installs the driver silently
- enables persistence mode
- runs `nvidia-smi`
- installs Poetry
- appends Poetry to the shell path for `azureuser`
- changes `/mnt` ownership to `azureuser`

### Step 4: verify the driver

Run:

```bash
nvidia-smi
```

You should see the NVIDIA A10 device exposed correctly. If `nvidia-smi` fails, the driver setup is not complete.

## Suggested Setup Sequence

1. Provision the VM with Ubuntu LTS and the `Standard_NV12ads_A10_v5` size.
2. SSH into the machine.
3. Choose one GPU setup path:
   - preferred: Azure-supported NVIDIA driver extension
   - fallback: repo manual two-part install scripts
4. If using the repo manual path:

```bash
bash scripts/azure/install_nvidia_part1.sh
```

5. Reconnect after reboot and verify the kernel:

```bash
uname -r
```

6. Finish the manual driver install:

```bash
bash scripts/azure/install_nvidia_part2.sh
```

7. Verify GPU visibility with:

```bash
nvidia-smi
```

8. Perform the common machine bootstrap:

```bash
sudo apt install make
ssh-keygen -t ed25519 -C "ec2-medwav" -f ~/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

git config --global user.email "giannisginis53@gmail.com"
git config --global user.name "Ioannis Gkinis"
sudo chown azureuser:azureuser /mnt
git clone git@github.com:hcmr-ai/Med-WAV.git
```

At this point, add the printed public key to GitHub if SSH clone access is not already configured.

9. Install repo dependencies if part 2 did not already handle Poetry in the expected shell context:

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
cd Med-WAV
poetry install
```

10. Configure blob access and mounts if needed:

```bash
./scripts/azure/setup_blobfuse_mounts.sh
```

11. Verify required mount points and data paths.
12. Run training with a derived config or an explicit intended config.

## Blob And Checkpoint Support

Relevant Azure helper scripts in this repo:
- [`scripts/azure/setup_blobfuse_mounts.sh`](../scripts/azure/setup_blobfuse_mounts.sh): mount `medwav-data` and `scalers`
- [`scripts/azure/download_checkpoint.py`](../scripts/azure/download_checkpoint.py): pull checkpoint folders from Azure Blob
- [`scripts/azure/install_nvidia_part1.sh`](../scripts/azure/install_nvidia_part1.sh): kernel preparation and reboot for the pinned manual NVIDIA path
- [`scripts/azure/install_nvidia_part2.sh`](../scripts/azure/install_nvidia_part2.sh): pinned manual NVIDIA GRID driver install and basic machine bootstrap
- [`scripts/azure/delete_a_blob.py`](../scripts/azure/delete_a_blob.py): destructive maintenance helper
- [`scripts/azure/delete_soft_deleted_blobs.py`](../scripts/azure/delete_soft_deleted_blobs.py): destructive maintenance helper

Handover note:
- the delete scripts are operational cleanup utilities, not normal workflow scripts
- they require `AZURE_STORAGE_CONNECTION_STRING`
- they are destructive and should not be run casually

The NVIDIA scripts are setup scripts, not full environment provisioning scripts. They do not:
- clone the repo
- validate all dataset paths
- mount Azure blob storage
- verify the active training config end to end

## Verification Checklist

After setup, verify:
- `nvidia-smi` works
- `poetry --version` works
- `mountpoint -q /mnt/blobstorage`
- `mountpoint -q /mnt/blobstorage-scalers`
- the configured data path exists
- the configured checkpoint/log roots are writable

Useful commands:

```bash
uname -r
poetry --version
mountpoint -q /mnt/blobstorage && echo ok
mountpoint -q /mnt/blobstorage-scalers && echo ok
```

Also verify:

```bash
git config --global user.email
git config --global user.name
ssh -T git@github.com
```

## Troubleshooting

### Wrong kernel after reboot

Symptom:
- `uname -r` is not `6.8.0-1025-azure`

Action:
- do not run part 2
- inspect GRUB/default boot selection
- rerun part 1 or correct the boot target manually

### `nvidia-smi` fails after part 2

Possible causes:
- wrong kernel
- driver installer failed silently
- installer URL changed or downloaded the wrong file
- Secure Boot / vTPM interfered with unsigned module loading

Checks:

```bash
uname -r
lsmod | grep nvidia
which nvidia-smi
```

### NVIDIA download URL is stale

The manual path hardcodes a Microsoft-hosted NVIDIA installer URL. If it expires or changes:
- check the current Microsoft Learn Linux N-series driver setup page
- update the doc first
- then update the script if the new URL is validated

### `/mnt` ownership is wrong

The manual part 2 script currently runs:

```bash
sudo chown azureuser:azureuser /mnt
```

If your active username is not `azureuser`, this assumption is wrong and should be corrected before relying on the script for a new environment.

### Git clone over SSH fails

Checks:
- confirm the public key from `~/.ssh/id_ed25519.pub` was added to GitHub
- confirm the SSH agent has the key loaded:

```bash
ssh-add -l
```

- test GitHub SSH access:

```bash
ssh -T git@github.com
```

## Practical Warnings

- The repo’s checked-in DNN config is tied to specific `/mnt` paths and an existing experiment layout.
- The manual NVIDIA scripts are intentionally pinned and may age badly as Azure host kernels and driver branches move.
- If the official extension path changes, prefer updating this document and the provisioning procedure before modifying the pinned scripts.
