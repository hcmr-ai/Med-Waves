from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from comet_ml import ExistingExperiment, Experiment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


class RandomDeltaSampling:
    def __init__(self, seed: int = 42, comet_experiment: Experiment = None, log_plots_per_file: bool = False, run_id: int = 0):
        self.seed = seed
        self.run_id = run_id
        self.error_distribution = {}
        self.log_plots_per_file = log_plots_per_file
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

        self.comet.log_parameter("seed", self.seed)
        self.comet.log_parameter("run_id", self.run_id)
        self.comet.add_tag(f"run_{self.run_id}")

    # def _log_histogram_old(self, values, label, bins=100):
    #     import matplotlib.pyplot as plt

    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    #     ax.set_title(f"Error Distribution: {label}")
    #     ax.set_xlabel("Error (target - input)")
    #     ax.set_ylabel("Frequency")
    #     ax.grid(True)

    #     self.comet.log_figure(figure_name=f"error_hist_{label}", figure=fig)
    #     plt.close(fig)

    def _log_histogram(self, values, label: str, bins: int = 100):
        import matplotlib.pyplot as plt
        print(f"📊 Logging error histogram for '{label}'...")

        # Wrap values in Polars for fast binning
        df = pl.DataFrame({"value": values})
        min_val, max_val = df["value"].min(), df["value"].max()
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Digitize manually
        bin_indices = np.digitize(df["value"].to_numpy(), bin_edges) - 1
        counts = np.bincount(bin_indices, minlength=bins)

        # Compute bin centers
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            bin_centers,
            counts[:len(bin_centers)],  # Ensure alignment
            width=bin_width,
            edgecolor='black',
            alpha=0.7,
            rasterized=True
        )
        ax.set_title(f"Error Distribution: {label}")
        ax.set_xlabel("Error (target - input)")
        ax.set_ylabel("Frequency")
        ax.grid(True)

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

        self.comet.log_metric(f"{label}_n_errors", len(all_errors))
        self._log_histogram(all_errors, label)

    def save_error_distribution(self, base_dir: str):
        """Save each error distribution as a separate Parquet file in the given directory."""
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving error distributions to directory: {base_dir}")

        for label, errors in self.error_distribution.items():
            df = pl.DataFrame({
                "error": pl.Series(errors)
            })
            file_path = base_dir / f"{label}.parquet"
            df.write_parquet(file_path)
            print(f"  ✅ Saved {len(errors)} errors for '{label}' → {file_path}")

    def load_error_distribution(self, base_dir: str, log_to_comet: bool = True):
        """Load error distributions from individual Parquet files in a directory."""
        base_dir = Path(base_dir)
        print(f"📂 Loading error distributions from directory: {base_dir}")

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {base_dir}")

        files = sorted(base_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {base_dir}")

        for file in files:
            label = file.stem  # filename without .parquet
            df = pl.read_parquet(file)
            if "error" not in df.columns:
                print(f"⚠️ Skipping {file.name}: no 'error' column.")
                continue
            errors = df["error"].to_numpy()
            self.error_distribution[label] = errors
            print(f"✅ Loaded {len(errors)} errors for label '{label}'")

            if log_to_comet:
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
        target_cols: dict,
        sample_fraction: float = 0.05
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.comet.log_text("eval_files:\n" + "\n".join(str(f) for f in eval_files))

        print(f"🚀 Predicting on {len(eval_files)} files...")

        # all_preds = {col: [] for _, _, col in predictions}
        # all_truths = {col: [] for col in target_cols.values()} if target_cols else {}
        # all_inputs = {
        #     **{col: [] for col, _, _ in predictions}
        # }
        # full_eval_df = []

        for file in tqdm(eval_files, desc="Predicting"):
            df = pl.read_parquet(file)
            df = df.drop_nulls(subset=["VHM0", "corrected_VHM0", "VTM02", "corrected_VTM02"])

            new_cols = []

            for input_col, label, output_col in predictions:
                pred = self.predict(df, input_col, label)

                # n_clipped = np.sum(np.array(pred) < 0)

                # self.comet.log_metric(f"{output_col}_n_clipped", int(n_clipped))

                new_cols.append(pl.Series(output_col, pred))
                # all_preds[output_col].extend(pred)

            df = df.with_columns(new_cols)

            # full_eval_df.append(df.select([
            #     "latitude", "longitude",
            #     *[c for c in df.columns if c.startswith("predicted_")],
            #     *[c for c in df.columns if c.startswith("corrected_")]
            # ]))

            # for _, target_col in target_cols.items():
            #     if target_col in df.columns:
            #         all_truths[target_col].extend(df[target_col].to_numpy())
            # for input_col, _, _ in predictions:
            #     if input_col in df.columns:
            #         all_inputs[input_col].extend(df[input_col].to_numpy())

            pred_output_dir = Path(output_dir) / str(self.run_id)
            pred_output_dir.mkdir(parents=True, exist_ok=True)

            df.write_parquet(pred_output_dir / file.name)

            # if self.log_plots_per_file:
            #     self._log_prediction_distribution(
            #         predictions={out: all_preds[out][-len(df):] for _, _, out in predictions},
            #         inputs={out: all_inputs[inp][-len(df):] for inp, _, out in predictions},
            #         targets={out: all_truths[tgt][-len(df):] for out, tgt in target_cols.items()},
            #         file_name=file.name.replace(".parquet", "")
            #     )
            #     self._log_metric_maps(
            #         df=df,
            #         lat_col="latitude",
            #         lon_col="longitude",
            #         prediction_target_pairs=[
            #             (out, tgt) for out, tgt in target_cols.items()
            #         ],
            #         file_prefix=file.name.replace(".parquet", "") + "_"
            #     )

        self.comet.add_tag("prediction")
        self.comet.add_tag("sampled_error")
        self.comet.add_tag("bias_correction")

        # clear
        del df, new_cols
        self.error_distribution.clear()

        # self._compute_and_log_metrics(all_truths, all_preds)

        # sampled_preds, sampled_inputs, sampled_targets = {}, {}, {}

        # for input_col, _, output_col in predictions:
        #     tgt_col = target_cols[output_col]

        #     # Create a Polars DataFrame
        #     df = pl.DataFrame({
        #         "input": all_inputs[input_col],
        #         "pred": all_preds[output_col],
        #         "target": all_truths[tgt_col]
        #     })

        #     # Sample a fraction (with replacement=False)
        #     sampled = df.sample(fraction=sample_fraction, with_replacement=False, seed=self.seed)

        #     # Store results
        #     sampled_preds[output_col] = sampled["pred"].to_list()
        #     sampled_inputs[output_col] = sampled["input"].to_list()
        #     sampled_targets[output_col] = sampled["target"].to_list()


        # del all_truths, all_inputs, all_preds

        # self._log_prediction_distribution(
        #     predictions=sampled_preds,
        #     inputs=sampled_inputs,
        #     targets=sampled_targets,
        #     file_name="aggregated_sampled"
        # )
        # df_concat = pl.concat(full_eval_df, how="vertical")
        # del full_eval_df

        # self._log_metric_maps(
        #     df=df_concat,
        #     lat_col="latitude",
        #     lon_col="longitude",
        #     prediction_target_pairs=[
        #         ("predicted_VHM0", "corrected_VHM0"),
        #         ("predicted_VTM02", "corrected_VTM02")
        #     ],
        #     metrics=["rmse", "mae", "bias"],
        #     file_prefix="aggregated__",
        #     grid_resolution=0.25,  # or any resolution you prefer
        #     cmap="viridis"
        # )

    def _compute_and_log_metrics_slow(self, y_true_dict, y_pred_dict):
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

            for k, v in metrics.items():
                self.comet.log_metric(f"{pred_col}_{k}", v)

    def _compute_and_log_metrics(self, y_true_dict, y_pred_dict):
        print("📈 Computing and logging evaluation metrics")

        for pred_col, preds in y_pred_dict.items():
            target_col = pred_col.replace("predicted_", "corrected_")
            if target_col not in y_true_dict:
                continue

            df = pl.DataFrame({
                "y_true": y_true_dict[target_col],
                "y_pred": preds
            }).with_columns([
                (pl.col("y_pred").clip(0.0, None)).alias("y_pred_clipped")
            ])

            # Compute errors
            df = df.with_columns([
                (pl.col("y_pred") - pl.col("y_true")).alias("diff"),
                (pl.col("y_pred_clipped") - pl.col("y_true")).alias("diff_clipped"),
            ])

            # Compute metrics using Polars expressions
            metrics = {
                "mae": df.select(pl.col("diff").abs().mean()).item(),
                "mae_clipped": df.select(pl.col("diff_clipped").abs().mean()).item(),
                "rmse": df.select((pl.col("diff") ** 2).mean().sqrt()).item(),
                "rmse_clipped": df.select((pl.col("diff_clipped") ** 2).mean().sqrt()).item(),
                "bias": df.select(pl.col("diff").mean()).item(),
            }

            # R2 still needs NumPy
            try:
                from sklearn.metrics import r2_score
                metrics["r2"] = r2_score(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
            except Exception:
                metrics["r2"] = float("nan")

            for name, val in metrics.items():
                self.comet.log_metric(f"{pred_col}_{name}", val)


    def _log_prediction_distribution(
        self,
        predictions: dict[str, list[float]],
        inputs: dict[str, list[float]],
        targets: dict[str, list[float]],
        file_name: str
    ):
        print("⚡ Fast histogram generation using Polars")
        import matplotlib.pyplot as plt

        for col, pred_values in predictions.items():
            input_values = inputs.get(col, [])
            target_values = targets.get(col, [])

            # Combine all values to compute consistent bin edges
            all_vals = np.concatenate([pred_values, input_values, target_values])
            min_val, max_val = float(np.min(all_vals)), float(np.max(all_vals))
            bins = np.linspace(min_val, max_val, 101)  # 100 bins

            def compute_hist(data: list[float], label: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                df = pl.DataFrame({"value": data})
                bin_indices = np.digitize(df["value"].to_numpy(), bins) - 1
                counts = np.bincount(bin_indices, minlength=len(bins) - 1)[:len(bins) - 1]
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                return bin_centers, counts

            fig, ax = plt.subplots(figsize=(8, 4))

            for values, label, color in zip(
                [input_values, pred_values, target_values],
                ["Input", "Predicted", "Target"],
                ["blue", "green", "orange"], strict=False
            ):
                if not values:
                    continue
                x, y = compute_hist(values, label, bins)
                ax.bar(x, y, width=(x[1] - x[0]), alpha=0.3, label=f"{label}", edgecolor='black', color=color)

            ax.set_title(f"Prediction Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True)
            ax.legend()

            self.comet.log_figure(
                figure_name=f"prediction_hists__{file_name}__{col}.png",
                figure=fig,
            )
            plt.close(fig)

    def _log_metric_maps(
        self,
        df: pl.DataFrame,
        lat_col: str,
        lon_col: str,
        prediction_target_pairs: list[tuple[str, str]],
        metrics: list[str] | None = None,
        file_prefix: str = "",
        grid_resolution: float = 0.01,
        cmap: str = "viridis"
    ):
        if metrics is None:
            metrics = ["rmse", "mae", "bias"]
        print("Plotting metric maps...")
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        def bin_coords(df, lat_col, lon_col, res):
            return df.with_columns([
                (pl.col(lat_col) / res).round(0) * res,
                (pl.col(lon_col) / res).round(0) * res
            ]).rename({lat_col: "lat_bin", lon_col: "lon_bin"})

        def compute_metric_expr(metric, pred_col, target_col):
            if metric == "rmse":
                return ((pl.col(pred_col) - pl.col(target_col)) ** 2).mean().sqrt().alias("metric")
            elif metric == "mae":
                return (pl.col(pred_col) - pl.col(target_col)).abs().mean().alias("metric")
            elif metric == "bias":
                return (pl.col(pred_col) - pl.col(target_col)).mean().alias("metric")
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        df = bin_coords(df, lat_col, lon_col, grid_resolution)

        for pred_col, target_col in prediction_target_pairs:
            for metric in metrics:
                grouped = df.group_by(["lat_bin", "lon_bin"]).agg([
                    compute_metric_expr(metric, pred_col, target_col)
                ])

                lats = grouped["lat_bin"].to_numpy()
                lons = grouped["lon_bin"].to_numpy()
                values = grouped["metric"].to_numpy()

                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()

                scatter = ax.scatter(
                    lons, lats, c=values, cmap=cmap, s=10, alpha=0.8, transform=ccrs.PlateCarree(), rasterized=True
                )

                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.set_title(f"{metric.upper()} Map for {pred_col}")
                plt.colorbar(scatter, ax=ax, label=metric.upper(), shrink=0.6)
                plt.tight_layout()

                self.comet.log_figure(
                    figure_name=f"{file_prefix}map__{metric}__{pred_col}.png",
                    figure=fig
                )
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


def main(run_id: int):
    load_from_parquet = True
    plot_from_logged = False

    output_dir = "/data/tsolis/AI_project/output/experiments/random_regressor"
    train_dir = Path("/data/tsolis/AI_project/parquet/augmented_with_labels/hourly")

    if plot_from_logged:
        experiment = ExistingExperiment(
            api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
            previous_experiment="1575e3591c534bc5841bc7cfba07d10c",
        )
        reg = RandomDeltaSampling(comet_experiment=experiment)
        reg.log_saved_predictions_to_comet(output_dir)
        exit()

    import random
    reg = RandomDeltaSampling(seed=random.randint(0, 1000000), log_plots_per_file=False, run_id=run_id)

    if load_from_parquet:
        reg.load_error_distribution(f"{output_dir}/error_distribution/sub_set_3_months")
    else:
        patterns = ["WAVEAN20211", "WAVEAN20221"]
        reg.comet.log_parameter("train_file_patterns", ", ".join(patterns))
        train_files = sorted(
            file for pattern in patterns for file in train_dir.glob(f"{pattern}*.parquet")
        )
        reg.fit_many(train_files, input_col="VHM0", target_col="corrected_VHM0", label="swh")
        reg.fit_many(train_files, input_col="VTM02", target_col="corrected_VTM02", label="mwp")
        reg.save_error_distribution(f"{output_dir}/error_distribution")

    patterns = ["WAVEAN2023"]
    reg.comet.log_parameter("eval_file_patterns", ", ".join(patterns))
    eval_files = sorted(
        file for pattern in patterns for file in train_dir.glob(f"*{pattern}*.parquet")
    )

    # Predict and save
    reg.batch_predict_and_save(
        eval_files,
        output_dir=output_dir,
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
    for i in range(3, 5):
        print(f'Running iteration {i}')
        main(i)
