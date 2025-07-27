from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from comet_ml import ExistingExperiment, Experiment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


class SampledErrorRegressor:
    def __init__(self, seed: int = 42, comet_experiment: Experiment = None):
        self.seed = seed
        self.error_distribution = {}

        if comet_experiment is None:
            self.comet = Experiment(
                api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                project_name="hcmr-ai",
                workspace="ioannisgkinis"
            )
            exp_name = f"random_regressor_bias_correction_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.comet.set_name(exp_name)
        else:
            self.comet = comet_experiment

    def _log_histogram(self, values, label, bins=100):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(f"Error Distribution: {label}")
        ax.set_xlabel("Error (target - input)")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        if self.comet:
            self.comet.log_figure(figure_name=f"error_hist_{label}", figure=fig)
        plt.close(fig)

    def fit_many(self, files, input_col: str, target_col: str, label: str):
        self.comet.log_text("train_files:\n" + "\n".join(str(f) for f in files))
        np.random.seed(self.seed)
        errors = []

        for file in tqdm(files, desc=f"Fitting {label}"):
            df = pl.read_parquet(file)
            df = df.drop_nulls(subset=["VHM0", "corrected_VHM0"])
            df = df.drop_nulls(subset=["VTM02", "corrected_VTM02"])
            if input_col not in df.columns or target_col not in df.columns:
                continue
            delta = (df[target_col] - df[input_col]).to_numpy()
            errors.append(delta)

        all_errors = np.concatenate(errors)
        self.error_distribution[label] = all_errors
        print(f"✅ Learned {len(all_errors)} errors for '{label}'")

        if self.comet:
            self.comet.log_metric(f"{label}_n_errors", len(all_errors))
            self._log_histogram(all_errors, label)

    def fit_stratified(self, files, input_col, target_col, bin_col, label, bins):
        binned_errors = {}

        for file in tqdm(files, desc=f"Fitting stratified {label} by {bin_col}"):
            df = pl.read_parquet(file)
            if input_col not in df.columns or target_col not in df.columns or bin_col not in df.columns:
                continue

            delta = (df[target_col] - df[input_col]).to_numpy()
            bin_vals = df[bin_col].to_numpy()
            bin_indices = np.digitize(bin_vals, bins)

            for i, b in enumerate(bin_indices):
                key = f"{label}_bin_{b}"
                if key not in binned_errors:
                    binned_errors[key] = []
                binned_errors[key].append(delta[i])

        for key, values in binned_errors.items():
            self.error_distribution[key] = np.array(values)
            if self.comet:
                self.comet.log_metric(f"{key}_n_errors", len(values))
                self._log_histogram(values, key)

    def save_error_distribution(self, path: str):
        """Efficiently save error distributions to a Parquet file."""
        dfs = []

        for label, errors in self.error_distribution.items():
            n = len(errors)
            df = pl.DataFrame({
                "label": pl.Series([label] * n, dtype=pl.Utf8),
                "error": pl.Series(errors)
            })
            dfs.append(df)

        full_df = pl.concat(dfs, how="vertical")
        full_df.write_parquet(path)
        print(f"💾 Saved error distribution to {path}")

    def load_error_distribution(self, path: str):
        """Load error distributions from a Parquet file."""
        print("Loading Error Distributions From Disk")
        df = pl.read_parquet(path)
        for label in df["label"].unique():
            # lbl = label.item()
            errors = df.filter(pl.col("label") == label)["error"].to_numpy()
            self.error_distribution[label] = errors
            print(f"✅ Loaded {len(errors)} errors for label '{label}'")

            if self.comet:
                self.comet.log_metric(f"{label}_n_loaded_errors", len(errors))
                self._log_histogram(errors, label)

    def predict(self, df: pl.DataFrame, input_col: str, label: str) -> np.ndarray:
        if label not in self.error_distribution:
            raise ValueError(f"No error distribution for label '{label}'")

        n = len(df)
        sampled_errors = np.random.choice(self.error_distribution[label], size=n)
        return df[input_col].to_numpy() + sampled_errors

    def batch_predict_and_save(
        self,
        eval_files,
        output_dir: str,
        predictions: list[tuple[str, str, str]],
        target_cols: dict = None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.comet.log_text("eval_files:\n" + "\n".join(str(f) for f in eval_files))

        print(f"🚀 Predicting on {len(eval_files)} files...")

        all_preds = {col: [] for _, _, col in predictions}
        all_truths = {col: [] for col in target_cols.values()} if target_cols else {}

        for file in tqdm(eval_files, desc="Predicting"):
            df = pl.read_parquet(file)
            df = df.drop_nulls(subset=["VHM0", "corrected_VHM0", "VTM02", "corrected_VTM02"])

            new_cols = []

            for input_col, label, output_col in predictions:
                pred = self.predict(df, input_col, label)

                n_clipped = np.sum(np.array(pred) < 0)

                if self.comet:
                    self.comet.log_metric(f"{output_col}_n_clipped", int(n_clipped))

                new_cols.append(pl.Series(output_col, pred))
                all_preds[output_col].extend(pred)

            df = df.with_columns(new_cols)

            if target_cols:
                for _, target_col in target_cols.items():
                    if target_col in df.columns:
                        all_truths[target_col].extend(df[target_col].to_numpy())

            df.write_parquet(f"{output_dir}/{file.name}")


        if target_cols and self.comet:
            self.comet.add_tag("prediction")
            self.comet.add_tag("sampled_error")
            self.comet.add_tag("bias_correction")

            self._compute_and_log_metrics(all_truths, all_preds)
            self._log_prediction_distribution(
                predictions=all_preds,
                originals={f"predicted_{k}": all_truths[f"corrected_{k}"] for k in ["VHM0", "VTM02"]}
            )

    def _compute_and_log_metrics(self, y_true_dict, y_pred_dict):
        print("Computing and logging Eval Metrics")
        for pred_col, y_pred in y_pred_dict.items():
            target_col = pred_col.replace("predicted_", "corrected_")
            if target_col not in y_true_dict:
                continue
            y_true = np.array(y_true_dict[target_col])
            y_pred = np.array(y_pred)

            metrics = {
                "mae": mean_absolute_error(y_true, y_pred),
                "mae_clipped": mean_absolute_error(y_true, np.clip(y_pred, 0, None)),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "rmse_clipped": np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 0, None))),
                "bias": np.mean(y_pred - y_true),
                "r2": r2_score(y_true, y_pred),
            }

            if self.comet:
                for k, v in metrics.items():
                    self.comet.log_metric(f"{pred_col}_{k}", v)

    def _log_prediction_distribution(
        self,
        predictions: dict[str, list[float]],
        originals: dict[str, list[float]],
        file_name: str
    ):
        print("Computing and logging Eval Plots")
        import matplotlib.pyplot as plt

        for col, pred_values in predictions.items():
            fig, ax = plt.subplots(figsize=(8, 4))

            all_vals = pred_values + (originals[col] if originals and col in originals else [])
            bins = np.histogram_bin_edges(all_vals, bins=100)
            # bins = 100

            # Plot predicted (corrected)
            ax.hist(pred_values, bins=bins, alpha=0.6, label="Predicted", edgecolor='black')

            # Optionally plot original (uncorrected)
            if col in originals:
                orig_values = originals[col]
                ax.hist(orig_values, bins=bins, alpha=0.6, label="Original", edgecolor='black')

            ax.set_title(f"Prediction Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True)
            ax.legend()

            if self.comet:
                self.comet.log_figure(figure_name=f"prediction_hist_{col}_{file_name}", figure=fig)

            plt.close(fig)

    def log_saved_predictions_to_comet(self, prediction_dir: str):
        prediction_dir = Path(prediction_dir)
        files = sorted(prediction_dir.glob("WAVEAN20231*.parquet"))

        print(f"📂 Reloading predictions from {len(files)} files...")

        for file in tqdm(files, desc="Reloading"):
            df = pl.read_parquet(file)

            # Check for necessary columns
            required_cols = ["VHM0", "VTM02", "predicted_VHM0", "predicted_VTM02"]
            if not all(c in df.columns for c in required_cols):
                print(f"⚠️ Skipping {file.name}: missing columns.")
                continue

            # Drop nulls
            df = df.drop_nulls(subset=required_cols)

            # Prepare per-file dictionaries
            per_file_preds = {
                "predicted_VHM0": df["predicted_VHM0"].to_list(),
                "predicted_VTM02": df["predicted_VTM02"].to_list()
            }
            per_file_originals = {
                "predicted_VHM0": df["VHM0"].to_list(),
                "predicted_VTM02": df["VTM02"].to_list()
            }

            # Log histograms just for this file
            print(f"📊 Logging histograms for {file.name} to Comet...")
            self._log_prediction_distribution(per_file_preds, per_file_originals, file.name)


def main():
    load_from_parquet = False
    plot_from_logged = True

    output_dir = "/data/tsolis/AI_project/output/experiments/random_regressor"

    if plot_from_logged:
        experiment = ExistingExperiment(
            api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
            previous_experiment="1575e3591c534bc5841bc7cfba07d10c",
        )
        reg = SampledErrorRegressor(comet_experiment=experiment)
        reg.log_saved_predictions_to_comet(output_dir)
        exit()

    reg = SampledErrorRegressor(seed=42)
    if load_from_parquet:
        reg.load_error_distribution(f"{output_dir}/error_distribution.parquet")
    else:
        train_dir = Path("/data/tsolis/AI_project/parquet/augmented_with_labels/hourly")
        patterns = ["WAVEAN2021", "WAVEAN2022"]
        reg.comet.log_parameter("train_file_patterns", ", ".join(patterns))
        train_files = sorted(
            file for pattern in patterns for file in train_dir.glob(f"{pattern}*.parquet")
        )
        reg.fit_many(train_files, input_col="VHM0", target_col="corrected_VHM0", label="swh")
        reg.fit_many(train_files, input_col="VTM02", target_col="corrected_VTM02", label="mwp")
        reg.save_error_distribution(f"{output_dir}/error_distribution.parquet")

    patterns = ["WAVEAN2023"]
    reg.comet.log_parameter("eval_file_patterns", ", ".join(patterns))
    eval_files = sorted(
        file for pattern in patterns for file in train_dir.glob(f"*{pattern}*.parquet")
    )

    # Predict and save
    reg.batch_predict_and_save(
        eval_files,
        output_dir="/data/tsolis/AI_project/output/experiments/random_regressor",
        predictions=[
            ("VHM0", "swh", "predicted_VHM0"),
            ("VTM02", "mwp", "predicted_VTM02")
        ],
        target_cols={
            "predicted_VHM0": "corrected_VHM0",
            "predicted_VTM02": "corrected_VTM02"
        }
    )


if __name__ == "__main__":
    main()
