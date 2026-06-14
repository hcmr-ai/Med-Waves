# Environment Setup

## Baseline

The repo is managed with Poetry and targets Python `>=3.11,<3.15`.

## Expected Runtime Environment

The main DNN workflow is designed for a Linux-like machine with:
- GPU access for training
- mounted datasets under `/mnt/...`
- optional Azure blobfuse mounts for checkpoints, diagnostics, and scalers

The repo can run some utilities locally on macOS, but the full training and evaluation workflow assumes the mounted `/mnt` layout used by the project.

For the standard Azure training machine currently used by the project, see [`docs/azure-training-vm.md`](azure-training-vm.md).

## Key Dependencies

Operationally important packages:
- `torch`
- `lightning`
- `transformers`
- `comet-ml`
- `s3fs`
- `azure-storage-blob`
- `polars`
- `xarray`
- `scikit-learn`

## Before Running Training

Confirm:
- Poetry dependencies are installed
- dataset path in the config exists
- blob storage mounts exist if the config points to `/mnt/blobstorage` or `/mnt/blobstorage-scalers`
- GPU visibility is correct if using `training.accelerator: gpu`
- Comet credentials are available if experiment tracking is enabled

Example:

```bash
ls /mnt/blobstorage
ls /mnt/blobstorage-scalers
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

If Comet tracking is enabled in config, the machine also needs the usual Comet environment variables or login state, for example:
- `COMET_API_KEY`
- `COMET_WORKSPACE`
- `COMET_PROJECT_NAME`

For handover-safe operation, the repo now expects all three Comet values to come from the runtime environment when Comet is enabled rather than from checked-in YAML defaults.

## Common Operator Bootstrap

On a fresh training machine, the usual bootstrap also includes:

```bash
sudo apt install make
ssh-keygen -t ed25519 -C "ec2-medwav" -f ~/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

git config --global user.email "giannisginis53@gmail.com"
git config --global user.name "Ioannis Gkinis"
sudo chown azureuser:azureuser /mnt
cd /mnt
git clone git@github.com:hcmr-ai/Med-WAV.git
```

Notes:
- `cat ~/.ssh/id_ed25519.pub` is used so the public key can be added to GitHub before cloning over SSH.
- The SSH key comment `ec2-medwav` is the existing convention even though the current training machine is Azure.
- `sudo chown azureuser:azureuser /mnt` assumes the username is `azureuser`.
- If the VM username differs, update the ownership command accordingly.

## Install Poetry And Project Dependencies

After the repo has been cloned into `/mnt`, install Poetry if needed and then install project dependencies:

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/Med-WAV
poetry install
```

Useful checks:

```bash
poetry --version
poetry run python --version
poetry run python -c "import torch, lightning; print(torch.__version__, lightning.__version__)"
```

## Create A Local Training Copy Of The Preprocessed Tensors

Before training, copy the preprocessed tensor dataset from mounted blob storage to local disk:

```bash
mkdir -p /mnt/local_datasets/
rsync -a --info=progress2 "/mnt/blobstorage/preprocessed_extended_subsampled_step_5/" "/mnt/local_datasets/preprocessed_extended_subsampled_step_5/"
```

Why this is done:
- training reads many `.pt` files repeatedly across epochs
- `/mnt/blobstorage/...` is blobfuse-backed mounted storage
- `/mnt/local_datasets/...` gives faster and more predictable local-disk reads
- this improves dataloader throughput and training stability
- it can also reduce repeated remote storage access during long runs, which may lower storage/network cost

The checked-in DNN config is already aligned with this pattern:
- `data.data_path: /mnt/local_datasets/preprocessed_extended_subsampled_step_5/`

## Agent / Skill Setup

Repo-local skills live under [`skills/`](../skills). Install them into supported agent environments with:

```bash
./scripts/install_repo_skills.sh
```

Targets supported by that script:
- Codex
- Claude
- Cursor
