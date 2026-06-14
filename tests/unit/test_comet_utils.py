import pytest

from src.commons.comet_utils import (
    get_comet_api_key,
    get_comet_project_name,
    get_comet_workspace,
    resolve_comet_settings,
)


@pytest.fixture(autouse=True)
def clear_comet_env(monkeypatch):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.delenv("COMET_WORKSPACE", raising=False)
    monkeypatch.delenv("COMET_PROJECT_NAME", raising=False)


def test_resolve_comet_settings_reads_all_values_from_env(monkeypatch):
    monkeypatch.setenv("COMET_API_KEY", "api-key")
    monkeypatch.setenv("COMET_WORKSPACE", "workspace-name")
    monkeypatch.setenv("COMET_PROJECT_NAME", "project-name")

    resolved = resolve_comet_settings()

    assert resolved == {
        "api_key": "api-key",
        "workspace": "workspace-name",
        "project_name": "project-name",
        "project": "project-name",
    }


def test_resolve_comet_settings_ignores_config_fallbacks_when_env_exists(monkeypatch):
    monkeypatch.setenv("COMET_API_KEY", "env-api-key")
    monkeypatch.setenv("COMET_WORKSPACE", "env-workspace")
    monkeypatch.setenv("COMET_PROJECT_NAME", "env-project")

    resolved = resolve_comet_settings(
        workspace="config-workspace",
        project_name="config-project",
    )

    assert resolved["workspace"] == "env-workspace"
    assert resolved["project_name"] == "env-project"
    assert resolved["project"] == "env-project"


@pytest.mark.parametrize(
    ("env_var", "expected_message"),
    [
        ("COMET_API_KEY", "COMET_API_KEY is not set"),
        ("COMET_WORKSPACE", "COMET_WORKSPACE is not set"),
        ("COMET_PROJECT_NAME", "COMET_PROJECT_NAME is not set"),
    ],
)
def test_resolve_comet_settings_requires_each_env_var(
    monkeypatch, env_var, expected_message
):
    monkeypatch.setenv("COMET_API_KEY", "api-key")
    monkeypatch.setenv("COMET_WORKSPACE", "workspace-name")
    monkeypatch.setenv("COMET_PROJECT_NAME", "project-name")
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(RuntimeError, match=expected_message):
        resolve_comet_settings()


def test_resolve_comet_settings_allows_empty_result_when_comet_disabled():
    assert resolve_comet_settings(require_api_key=False) == {}


def test_direct_helpers_require_env_values(monkeypatch):
    monkeypatch.setenv("COMET_API_KEY", "api-key")
    monkeypatch.setenv("COMET_WORKSPACE", "workspace-name")
    monkeypatch.setenv("COMET_PROJECT_NAME", "project-name")

    assert get_comet_api_key() == "api-key"
    assert get_comet_workspace() == "workspace-name"
    assert get_comet_project_name() == "project-name"


def test_direct_helpers_can_be_marked_optional():
    assert get_comet_api_key(required=False) is None
    assert get_comet_workspace(required=False) is None
    assert get_comet_project_name(required=False) is None
