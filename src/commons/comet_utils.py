"""Shared helpers for resolving Comet ML credentials and metadata."""

from __future__ import annotations

import os
from typing import Any


def get_comet_api_key(required: bool = True) -> str | None:
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        return api_key
    if required:
        raise RuntimeError(
            "COMET_API_KEY is not set. Set it in the environment before enabling Comet ML."
        )
    return None


def get_comet_workspace(required: bool = True) -> str | None:
    workspace = os.getenv("COMET_WORKSPACE")
    if workspace:
        return workspace
    if required:
        raise RuntimeError(
            "COMET_WORKSPACE is not set. Set it in the environment before enabling Comet ML."
        )
    return None


def get_comet_project_name(required: bool = True) -> str | None:
    project_name = os.getenv("COMET_PROJECT_NAME")
    if project_name:
        return project_name
    if required:
        raise RuntimeError(
            "COMET_PROJECT_NAME is not set. Set it in the environment before enabling Comet ML."
        )
    return None


def resolve_comet_settings(
    *,
    project_name: str | None = None,
    workspace: str | None = None,
    require_api_key: bool = True,
) -> dict[str, Any]:
    """Resolve Comet settings strictly from environment variables.

    The ``project_name`` and ``workspace`` parameters are kept only for
    backward-compatible call signatures. Checked-in config values are ignored
    so handover-safe routing always comes from the runtime environment.
    """
    resolved: dict[str, Any] = {}

    api_key = get_comet_api_key(required=require_api_key)
    if api_key is not None:
        resolved["api_key"] = api_key

    resolved_workspace = get_comet_workspace(required=require_api_key)
    if resolved_workspace:
        resolved["workspace"] = resolved_workspace

    resolved_project = get_comet_project_name(required=require_api_key)
    if resolved_project:
        resolved["project_name"] = resolved_project
        resolved["project"] = resolved_project

    return resolved
