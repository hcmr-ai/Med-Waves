import glob
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import polars as pl
import yaml
from comet_ml import Experiment
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

from src.data_engineering.split import holdout_split
from src.evaluation.metrics import evaluate_model
from src.evaluation.visuals import plot_residual_distribution
from src.features.helpers import extract_features_from_file


class IncrementalTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.data_dir_x = config["data_dir_x"]
        self.data_dir_y = config["data_dir_y"]
        self.batch_size = config["batch_size"]
        self.use_dask = config["use_dask"]
        self.save_model = config["save_model"]
        self.save_path = config["save_path"]
        self.log_batch_metrics = config["log_batch_metrics"]

        self.model = SGDRegressor()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.fitted_scaler = False
        self.experiment = None
        self._setup_model_tracker(config)

    def _setup_model_tracker(self, config: Dict[str, Any]):
        if config.get("use_comet", False):
            experiment = Experiment(
                api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                project_name=config["comet_project"],
                workspace=config["comet_workspace"],
            )
            experiment.log_parameters(config)
            self.experiment = experiment

    def train(self, x_train_files: List[str], y_train_files: List[str]):
        for i in tqdm(range(0, len(x_train_files), self.batch_size)):
            batch_dfs = []
            for j in range(i, min(i + self.batch_size, len(x_train_files))):
                df = extract_features_from_file(
                    x_train_files[j], y_train_files[j], use_dask=self.use_dask
                )
                batch_dfs.append(df)

            batch = pl.concat(batch_dfs)
            X_raw = batch.select(["vhm0_x", "wspd", "lat", "lon"]).to_numpy()
            y_raw = batch["vhm0_y"].to_numpy()

            mask = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(y_raw)
            X = X_raw[mask]
            y = y_raw[mask]

            if not self.fitted_scaler:
                X_poly = self.poly.fit_transform(X)
                X_scaled = self.scaler.fit_transform(X_poly)
                self.fitted_scaler = True
            else:
                X_poly = self.poly.transform(X)
                X_scaled = self.scaler.transform(X_poly)

            self.model.partial_fit(X_scaled, y)

            if self.log_batch_metrics:
                y_pred = self._predict(X)
                metrics = evaluate_model(y_pred, y)
                print(f"Batch {i // self.batch_size + 1} Metrics:")
                for k, v in metrics.items():
                    print(f"{k.upper()}: {v}")
                    if self.experiment:
                        self.experiment.log_metric(f"train_{k.upper()}", v, step=i)

            if self.save_model:
                joblib.dump(self.model, self.save_path)

    def _predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

    def evaluate(self, x_test_files: List[str], y_test_files: List[str]):
        test_dfs = []
        for x_f, y_f in zip(x_test_files, y_test_files, strict=False):
            df = extract_features_from_file(x_f, y_f, use_dask=self.use_dask)
            test_dfs.append(df)

        test_batch = pl.concat(test_dfs)
        X_raw = test_batch.select(["vhm0_x", "wspd", "lat", "lon"]).to_numpy()
        y_raw = test_batch["vhm0_y"].to_numpy()

        mask = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(y_raw)
        X = X_raw[mask]
        y = y_raw[mask]

        y_pred = self._predict(X)

        metrics = evaluate_model(y_pred, y)

        print("\\nFinal Evaluation:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v}")
            if self.experiment:
                self.experiment.log_metric(f"eval_{k.upper()}", v)

        fig = plot_residual_distribution(y, y_pred, "residual_distibution")

        if self.experiment:
            self.experiment.log_figure(figure=fig, figure_name="residual_distribution")


def main():
    with open("src/configs/config.yaml", "r") as f:
        experiment_cfg = yaml.safe_load(f)

    trainer = IncrementalTrainer(experiment_cfg)

    x_files = sorted(glob.glob(os.path.join(trainer.data_dir_x, "*.nc")))
    y_files = sorted(glob.glob(os.path.join(trainer.data_dir_y, "*.nc")))

    assert len(x_files) == len(y_files), "Mismatch in number of X and Y files"

    x_train, y_train, x_test, y_test = holdout_split(x_files, y_files)

    trainer.train(x_train, y_train)
    trainer.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
