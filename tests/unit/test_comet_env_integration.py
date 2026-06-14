from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from src.evaluation.experiment_logger import ExperimentLogger
from src.pipelines.training import dnn_trainer
from src.pipelines.training.train_random_regressor import RandomDeltaSampling


class FakeExperiment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.disabled = False
        self.url = "https://example.com/exp"
        self.tags = []
        self.parameters = {}
        self.name = None

    def set_name(self, name):
        self.name = name

    def add_tag(self, tag):
        self.tags.append(tag)

    def log_parameter(self, key, value):
        self.parameters[key] = value

    def log_parameters(self, values, prefix=None):
        if prefix:
            self.parameters[prefix] = values
        else:
            self.parameters.update(values)

    def log_metric(self, *args, **kwargs):
        return None

    def end(self):
        return None

    def get_key(self):
        return "fake-key"

    def log_other(self, key, value):
        self.parameters[key] = value

    def log_asset(self, *args, **kwargs):
        return None


def _load_script_module(script_relative_path: str, module_name: str):
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / script_relative_path
    spec = spec_from_file_location(module_name, script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_random_delta_sampling_creates_comet_experiment_from_resolved_env(monkeypatch):
    captured = {}

    def fake_resolve_comet_settings(*, require_api_key, **_kwargs):
        captured["require_api_key"] = require_api_key
        return {
            "api_key": "api-key",
            "workspace": "workspace-name",
            "project_name": "project-name",
            "project": "project-name",
        }

    monkeypatch.setattr(
        "src.pipelines.training.train_random_regressor.resolve_comet_settings",
        fake_resolve_comet_settings,
    )
    monkeypatch.setattr(
        "src.pipelines.training.train_random_regressor.Experiment",
        FakeExperiment,
    )

    sampler = RandomDeltaSampling(seed=7, run_id=3)

    assert captured["require_api_key"] is True
    assert sampler.comet.kwargs == {
        "api_key": "api-key",
        "workspace": "workspace-name",
        "project_name": "project-name",
    }
    assert sampler.comet.parameters["seed"] == 7
    assert sampler.comet.parameters["run_id"] == 3
    assert "run_3" in sampler.comet.tags


def test_experiment_logger_uses_resolved_env_for_comet_init(monkeypatch):
    captured = {}

    def fake_resolve_comet_settings(*, require_api_key, **_kwargs):
        captured["require_api_key"] = require_api_key
        return {
            "api_key": "api-key",
            "workspace": "workspace-name",
            "project_name": "project-name",
            "project": "project-name",
        }

    monkeypatch.setattr(
        "src.evaluation.experiment_logger.resolve_comet_settings",
        fake_resolve_comet_settings,
    )
    monkeypatch.setattr("src.evaluation.experiment_logger.Experiment", FakeExperiment)

    logger = ExperimentLogger(
        {
            "logging": {"use_comet": True},
            "output": {"experiment_name": "integration-check"},
            "model": {"foo": "bar"},
            "data": {"train_year": 2021},
        }
    )

    assert captured["require_api_key"] is True
    assert logger.experiment is not None
    assert logger.experiment.kwargs["api_key"] == "api-key"
    assert logger.experiment.kwargs["workspace"] == "workspace-name"
    assert logger.experiment.kwargs["project_name"] == "project-name"
    assert logger.experiment.kwargs["auto_param_logging"] is False
    assert logger.experiment.kwargs["auto_metric_logging"] is False
    assert logger.experiment.kwargs["experiment_name"].endswith("integration-check")
    assert logger.experiment.name is not None
    assert logger.experiment.name.endswith("integration-check")


def test_manual_experiment_script_uses_resolved_env(monkeypatch):
    module = _load_script_module(
        "scripts/create_manual_experiment.py",
        "create_manual_experiment_test_module",
    )
    captured = {}

    def fake_resolve_comet_settings(*, require_api_key, **_kwargs):
        captured["require_api_key"] = require_api_key
        return {
            "api_key": "api-key",
            "workspace": "workspace-name",
            "project_name": "project-name",
            "project": "project-name",
        }

    monkeypatch.setattr(module, "resolve_comet_settings", fake_resolve_comet_settings)
    monkeypatch.setattr(module, "Experiment", FakeExperiment)

    experiment = module.create_manual_experiment()

    assert captured["require_api_key"] is True
    assert experiment.kwargs == {
        "api_key": "api-key",
        "workspace": "workspace-name",
        "project_name": "project-name",
    }
    assert {"manual", "evaluation", "diff_corrector"} <= set(experiment.tags)
    assert experiment.parameters["evaluation_type"] == "manual"
    assert experiment.parameters["corrector"] == "DiffCorrector"


class FakeCometLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.experiment = FakeExperiment()


class FakeTensorBoardLogger:
    def __init__(self, save_dir, name):
        self.save_dir = save_dir
        self.name = name


def test_dnn_trainer_logger_creation_uses_resolved_env(monkeypatch):
    captured = {}

    class StubConfig:
        config = {
            "logging": {
                "use_comet": True,
                "use_tensorboard": True,
                "resolved_tensorboard_log_dir": "/tmp/tb",
                "log_dir": "/tmp/logs",
                "experiment_name": "dnn-comet-check",
                "comet_tags": ["handover"],
                "comet_project": "ignored-config-project",
                "comet_workspace": "ignored-config-workspace",
            },
            "data": {
                "train_year": 2021,
                "val_year": 2022,
                "target_columns": ["corrected_VHM0"],
                "predict_bias": True,
            },
        }

    def fake_resolve_comet_settings(*, require_api_key, **_kwargs):
        captured["require_api_key"] = require_api_key
        return {
            "api_key": "api-key",
            "workspace": "workspace-name",
            "project_name": "project-name",
            "project": "project-name",
        }

    def fake_log_training_artifacts(comet_logger, config_path):
        captured["artifact_config_path"] = config_path
        captured["artifact_logger"] = comet_logger

    monkeypatch.setattr(dnn_trainer, "resolve_comet_settings", fake_resolve_comet_settings)
    monkeypatch.setattr(dnn_trainer, "CometLogger", FakeCometLogger)
    monkeypatch.setattr(dnn_trainer, "TensorBoardLogger", FakeTensorBoardLogger)
    monkeypatch.setattr(dnn_trainer, "_log_training_artifacts", fake_log_training_artifacts)

    loggers, comet_logger, tensorboard_logger = dnn_trainer.create_experiment_loggers(
        StubConfig(), "configs/test-config.yaml"
    )

    assert captured["require_api_key"] is True
    assert captured["artifact_config_path"] == "configs/test-config.yaml"
    assert comet_logger.kwargs["api_key"] == "api-key"
    assert comet_logger.kwargs["workspace"] == "workspace-name"
    assert comet_logger.kwargs["name"] == "dnn-comet-check"
    assert comet_logger.kwargs["tags"] == ["handover"]
    assert tensorboard_logger.save_dir == "/tmp/tb"
    assert tensorboard_logger.name == "dnn-comet-check"
    assert loggers == [tensorboard_logger, comet_logger]
    assert comet_logger.experiment.parameters["training_data_year"] == 2021
    assert comet_logger.experiment.parameters["validation_data_year"] == 2022
    assert comet_logger.experiment.parameters["predict_bias"] is True
