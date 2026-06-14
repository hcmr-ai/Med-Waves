from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import xarray as xr
from IPython.display import display


DEFAULT_PATH_PATTERNS: dict[str, dict[str, str | None]] = {
    "azure": {
        "nc_raw_uncorrected": "/mnt/blobstorage/raw/without_reduced/WAVEAN{date}.nc",
        "nc_reference": "/mnt/blobstorage/raw/with_reduced/WAVEAN{date}.nc",
        "parquet_engineered": "/mnt/blobstorage/parquet/hourly_extra_features/year={year}/WAVEAN{date}.parquet",
        "pt_preprocessed": "/mnt/local_datasets/preprocessed_extended_subsampled_step_5/WAVEAN{date}.pt",
        "pt_preprocessed_blob": "/mnt/blobstorage/preprocessed_extended_subsampled_step_5/WAVEAN{date}.pt",
    },
    "neptune": {
        "nc_raw_uncorrected": "/data/tsolis/AI_project/without_reduced/WAVEAN{date}.nc",
        "nc_reference": "/data/tsolis/AI_project/with_reduced/WAVEAN{date}.nc",
        "parquet_engineered": "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN{date}.parquet",
        "pt_preprocessed": "/data/tsolis/AI_project/preprocessed_subsampled_step_5/WAVEAN{date}.pt",
        "pt_preprocessed_blob": None,
    },
}


PREFERRED_VARIABLES = [
    "VHM0",
    "corrected_VHM0",
    "VTM02",
    "corrected_VTM02",
    "WSPD",
]


ARTIFACT_DESCRIPTIONS = {
    "nc_raw_uncorrected": "Raw degraded model data (.nc)",
    "nc_reference": "Raw reference / corrected data (.nc)",
    "parquet_engineered": "Engineered modeling table (.parquet)",
    "pt_preprocessed": "Training-ready preprocessed tensor (.pt)",
    "pt_preprocessed_blob": "Blob-backed preprocessed tensor (.pt)",
}


@dataclass
class ArtifactView:
    artifact_key: str
    variable_name: str
    summary: dict[str, Any]
    preview: Any
    spatial_grid: np.ndarray | None
    histogram_values: np.ndarray | None


def build_artifact_paths(
    sample_date: str,
    env: str,
    root_overrides: dict[str, str | None] | None = None,
) -> dict[str, Path | None]:
    if env not in DEFAULT_PATH_PATTERNS:
        raise ValueError(f"Unsupported environment: {env}")

    year = sample_date[:4]
    patterns = dict(DEFAULT_PATH_PATTERNS[env])
    if root_overrides:
        patterns.update(root_overrides)

    resolved: dict[str, Path | None] = {}
    for key, pattern in patterns.items():
        if pattern is None:
            resolved[key] = None
        else:
            resolved[key] = Path(pattern.format(date=sample_date, year=year))
    return resolved


def locate_sample_files(
    sample_date: str,
    envs: tuple[str, ...] = ("azure", "neptune"),
    overrides_by_env: dict[str, dict[str, str | None]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overrides_by_env = overrides_by_env or {}

    for env in envs:
        paths = build_artifact_paths(sample_date, env, overrides_by_env.get(env))
        for artifact_key, path in paths.items():
            rows.append(
                {
                    "environment": env,
                    "artifact_key": artifact_key,
                    "description": ARTIFACT_DESCRIPTIONS.get(artifact_key, artifact_key),
                    "path": str(path) if path else None,
                    "exists": bool(path and path.exists()),
                }
            )

    return pd.DataFrame(rows)


def find_first_available_artifact(
    sample_date: str,
    env: str,
    artifact_priority: list[str],
    root_overrides: dict[str, str | None] | None = None,
) -> tuple[str, Path]:
    paths = build_artifact_paths(sample_date, env, root_overrides)
    for artifact_key in artifact_priority:
        path = paths.get(artifact_key)
        if path and path.exists():
            return artifact_key, path
    raise FileNotFoundError(
        f"No available artifact for sample {sample_date} in {env}. "
        f"Checked: {artifact_priority}"
    )


def choose_variable(candidates: list[str]) -> str:
    candidates_lower = {candidate.lower(): candidate for candidate in candidates}
    for name in PREFERRED_VARIABLES:
        if name in candidates:
            return name
        if name.lower() in candidates_lower:
            return candidates_lower[name.lower()]
    return candidates[0]


def _first_2d_slice(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D slice after squeezing, got shape {arr.shape}")
    return arr


def _finite_histogram_values(values: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    return flat if flat.size else None


def inspect_netcdf(path: str | Path, artifact_key: str) -> ArtifactView:
    ds = xr.open_dataset(path)
    try:
        variable_name = choose_variable(list(ds.data_vars))
        field = ds[variable_name]
        spatial_grid = _first_2d_slice(field.values)
        histogram_values = _finite_histogram_values(field.values)

        preview = ds[variable_name].to_dataframe().reset_index().head(10)
        summary = {
            "path": str(path),
            "dims": dict(ds.dims),
            "data_vars": list(ds.data_vars),
            "coords": list(ds.coords),
            "selected_variable": variable_name,
            "selected_variable_dims": list(field.dims),
        }
        return ArtifactView(
            artifact_key=artifact_key,
            variable_name=variable_name,
            summary=summary,
            preview=preview,
            spatial_grid=spatial_grid,
            histogram_values=histogram_values,
        )
    finally:
        ds.close()


def inspect_parquet(path: str | Path, artifact_key: str) -> ArtifactView:
    meta = pq.read_metadata(path)
    schema = meta.schema.to_arrow_schema()
    columns = schema.names
    variable_name = choose_variable(columns)

    table = pq.read_table(path)
    df = table.to_pandas()

    if "time" in df.columns:
        first_time = df["time"].dropna().iloc[0]
        df_plot = df[df["time"] == first_time].copy()
    else:
        df_plot = df.copy()

    spatial_grid = None
    if {"latitude", "longitude", variable_name}.issubset(df_plot.columns):
        pivot = (
            df_plot.pivot_table(
                index="latitude",
                columns="longitude",
                values=variable_name,
                aggfunc="mean",
            )
            .sort_index(ascending=False)
            .sort_index(axis=1)
        )
        if not pivot.empty:
            spatial_grid = pivot.to_numpy()

    histogram_values = (
        _finite_histogram_values(df[variable_name].to_numpy())
        if variable_name in df.columns
        else None
    )
    summary = {
        "path": str(path),
        "num_rows": meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "num_columns": len(columns),
        "columns": columns,
        "selected_variable": variable_name,
    }
    return ArtifactView(
        artifact_key=artifact_key,
        variable_name=variable_name,
        summary=summary,
        preview=df.head(10),
        spatial_grid=spatial_grid,
        histogram_values=histogram_values,
    )


def inspect_pt(path: str | Path, artifact_key: str) -> ArtifactView:
    data = torch.load(path, map_location="cpu", weights_only=False)
    tensor = data["tensor"]
    feature_cols = data["feature_cols"]
    variable_name = choose_variable(feature_cols)
    var_idx = feature_cols.index(variable_name)

    if tensor.ndim == 4:
        field_values = tensor[0, :, :, var_idx].numpy()
        feature_sample_values = tensor[0, 0, 0, :].numpy()
    elif tensor.ndim == 3:
        field_values = tensor[:, :, var_idx].numpy()
        feature_sample_values = tensor[0, 0, :].numpy()
    else:
        raise ValueError(f"Unexpected tensor shape for {path}: {tuple(tensor.shape)}")

    histogram_values = _finite_histogram_values(tensor[..., var_idx].numpy())
    summary = {
        "path": str(path),
        "tensor_shape": tuple(tensor.shape),
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "selected_variable": variable_name,
    }
    preview = pd.DataFrame(
        {
            "feature_cols": feature_cols,
            "sample_value_t0_y0_x0": feature_sample_values,
        }
    )
    return ArtifactView(
        artifact_key=artifact_key,
        variable_name=variable_name,
        summary=summary,
        preview=preview,
        spatial_grid=field_values,
        histogram_values=histogram_values,
    )


def inspect_artifact(path: str | Path, artifact_key: str) -> ArtifactView:
    suffix = Path(path).suffix.lower()
    if suffix == ".nc":
        return inspect_netcdf(path, artifact_key)
    if suffix == ".parquet":
        return inspect_parquet(path, artifact_key)
    if suffix == ".pt":
        return inspect_pt(path, artifact_key)
    raise ValueError(f"Unsupported artifact suffix: {suffix}")


def render_artifact_view(view: ArtifactView) -> None:
    print(f"Artifact: {view.artifact_key} ({ARTIFACT_DESCRIPTIONS.get(view.artifact_key, view.artifact_key)})")
    # print(f"Selected variable: {view.variable_name}")
    print()
    for key, value in view.summary.items():
        print(f"{key}: {value}")

    display(view.preview)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if view.spatial_grid is not None:
        # Use a Cartesian-style origin so spatial slices are not vertically flipped.
        im = axes[0].imshow(view.spatial_grid, aspect="auto", origin="lower")
        axes[0].set_title(f"Spatial slice: {view.variable_name}")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        axes[0].text(0.5, 0.5, "No spatial grid available", ha="center", va="center")
        axes[0].set_axis_off()

    if view.histogram_values is not None:
        axes[1].hist(view.histogram_values, bins=40)
        axes[1].set_title(f"Histogram: {view.variable_name}")
    else:
        axes[1].text(0.5, 0.5, "No finite values available", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def compare_artifacts(
    sample_date: str,
    env: str,
    artifact_keys: list[str],
    root_overrides: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    paths = build_artifact_paths(sample_date, env, root_overrides)
    rows: list[dict[str, Any]] = []

    for artifact_key in artifact_keys:
        path = paths.get(artifact_key)
        if not path or not path.exists():
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "path": str(path) if path else None,
                    "exists": False,
                }
            )
            continue

        view = inspect_artifact(path, artifact_key)
        rows.append(
            {
                "artifact_key": artifact_key,
                "path": str(path),
                "exists": True,
                "selected_variable": view.variable_name,
                "spatial_shape": None if view.spatial_grid is None else tuple(view.spatial_grid.shape),
                "preview_rows": len(view.preview),
            }
        )

    return pd.DataFrame(rows)
