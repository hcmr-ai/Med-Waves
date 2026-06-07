---
name: "medwav-poetry-full-install"
description: "Use when setting up Med-WAV from scratch with Poetry, especially when Poetry may be missing. This skill checks for Poetry, installs it if needed, exports Poetry to PATH for the current shell, installs the project environment, and verifies torch/CUDA in the Poetry virtualenv."
disable-model-invocation: true
---

# Med-WAV Poetry Full Install

Use this skill when the user asks for complete environment setup in this repo: check Poetry, install Poetry if missing, export PATH, install dependencies, and verify the environment.

Primary command sources:
- `Makefile` targets: `install-poetry`, `verify-poetry`, `install-dev`
- direct Poetry install: `poetry install --only main,test,dev --no-interaction`

## Default workflow

1. Confirm repo root and read current tool versions:

```bash
pwd
python3 --version
```

2. Check whether Poetry exists:

```bash
poetry --version
```

3. If Poetry is missing, install it (from `Makefile` behavior):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

4. Export Poetry into PATH for the current shell session (portable form):

```bash
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

5. Install environment dependencies (preferred full dev/test setup):

```bash
poetry install --only main,test,dev --no-interaction
```

6. Verify virtualenv and key runtime packages:

```bash
poetry env info -p
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Makefile mapping

When user asks to use Make targets, map as follows:
- Poetry bootstrap: `make install-poetry`
- Poetry check: `make verify-poetry`
- Dependencies: `make install-dev`
- Full setup target: `make setup-full`

## Important caveats

- `make setup-path` and `make install-poetry` print `/home/ubuntu/.local/bin`; on non-ubuntu usernames prefer `$HOME/.local/bin`.
- If shell PATH is not persisted, add it permanently:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

- If PyTorch CUDA wheels are pinned in `pyproject.toml` with the `torch-stable` source, do not run a separate `pip install torch ...` step unless troubleshooting.

## Troubleshooting checklist

- If `poetry` still not found after install, run `export PATH="$HOME/.local/bin:$PATH"` and retry.
- If dependency solve fails, run `poetry lock` and retry `poetry install`.
- If torch imports but CUDA is false, verify NVIDIA driver/CUDA runtime visibility on host.
