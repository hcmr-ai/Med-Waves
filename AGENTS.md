# Agent Instructions

Repository-specific agent instructions live here. Keep this file concise and treat it as the source of truth for coding agents working in this repo.

## Purpose

This repository trains and evaluates wave-correction models, including a documented DNN workflow:
- config: [`src/configs/config_dnn.yaml`](src/configs/config_dnn.yaml)
- training: [`src/pipelines/training/dnn_trainer.py`](src/pipelines/training/dnn_trainer.py)
- evaluation: [`src/pipelines/evaluation/evaluate_bunet.py`](src/pipelines/evaluation/evaluate_bunet.py)

## Preferred Workflow

Use these as the documented operator paths unless the user explicitly asks for another one:
- environment setup: [`docs/environment-setup.md`](docs/environment-setup.md)
- Azure training VM: [`docs/azure-training-vm.md`](docs/azure-training-vm.md)
- data, mounts, and server layout: [`docs/data-locations.md`](docs/data-locations.md)
- features: [`docs/features.md`](docs/features.md)
- preprocessing: [`docs/preprocessing-pipeline.md`](docs/preprocessing-pipeline.md)
- training architecture: [`docs/training-architecture.md`](docs/training-architecture.md)
- DNN training: [`docs/training-dnn.md`](docs/training-dnn.md)
- DNN evaluation: [`docs/evaluation_dnn.md`](docs/evaluation_dnn.md)
- baseline and alternative models: [`docs/baseline-models.md`](docs/baseline-models.md)
- config semantics: [`docs/config-reference.md`](docs/config-reference.md)

## Working Assumptions

- The repo is config-driven.
- Many workflows assume mounted storage under `/mnt/...`.
- Azure blob mounts are provisioned by [`scripts/azure/setup_blobfuse_mounts.sh`](scripts/azure/setup_blobfuse_mounts.sh).
- The documented preprocessing path excludes `scripts/helpers/` and `notebooks/`; treat those as legacy artifacts from a previous researcher unless the user explicitly asks about them.
- Training commonly assumes GPU and Poetry.
- The repo contains legacy or exploratory paths. Do not present them as the default unless confirmed.

## Safety Rules

- Do not casually rewrite hardcoded storage paths, checkpoint roots, or experiment naming unless the task is explicitly about infrastructure or experiment management.
- Treat `resume_from_checkpoint`, `checkpoint_dir`, `log_dir`, and `experiment_name` as coupled fields for experiment hygiene.
- Do not assume local data exists. Confirm mount or path assumptions from config before proposing commands.
- Treat secrets carefully. Do not echo storage keys or other credentials into logs or docs.
- Prefer derived configs or wrapper scripts for sweeps and one-off experiment changes instead of mutating the base config destructively.

## Validation Expectations

- For shell scripts: run `bash -n` and a dry-run or `--help` path when possible.
- For documentation changes: keep top-level docs aligned with the main recommended entrypoints.
- For config or pipeline changes: verify the documented commands still match the actual CLI.

## Agent-Specific Notes

- `CLAUDE.md` should remain a thin pointer back to this file to avoid duplicate instructions.
- Repo-local skills live under [`skills/`](skills). Install them with [`scripts/install_repo_skills.sh`](scripts/install_repo_skills.sh) if needed.
- Current repo skills:
  - [`medwav-blobfuse-mounts`](skills/medwav-blobfuse-mounts/SKILL.md) for Azure blobfuse mount setup and troubleshooting
  - [`medwav-poetry-full-install`](skills/medwav-poetry-full-install/SKILL.md) for full Poetry environment bootstrap and verification
