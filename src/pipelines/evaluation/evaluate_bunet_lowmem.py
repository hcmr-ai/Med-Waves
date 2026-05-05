#!/usr/bin/env python3
"""
Memory-efficient evaluation script for WaveBiasCorrector — full-grid safe.

Fixes two RAM bottlenecks in ModelEvaluator:

  1. spatial_errors_model / spatial_errors_baseline
     Original: list of per-batch (H,W) dicts  →  O(N_batches × H × W) RAM
     Here:     single in-place running sum      →  O(H × W) RAM

  2. plot_samples
     Original: unbounded Python lists → 350M+ elements on a full grid
     Here:     capped at --max-plot-samples (default 10_000_000) via
               per-batch random subsampling

All metrics, CSV outputs and plots are identical to evaluate_bunet.py.

Usage (drop-in replacement):
    poetry run python -m src.pipelines.evaluation.evaluate_bunet_lowmem \
        --config src/configs/config_dnn.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --output-dir ./evaluation_results \
        --max-plot-samples 10000000 \
        --save-predictions
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.evaluation.evaluate_bunet import ModelEvaluator, main as _base_main


class LowMemModelEvaluator(ModelEvaluator):

    def __init__(self, *args, max_plot_samples: int = 10_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_plot_samples = max_plot_samples
        self._spatial_model_accum: Optional[dict] = None
        self._spatial_baseline_accum: Optional[dict] = None

    # ------------------------------------------------------------------
    # 1. Spatial errors: drain list into running sum after each batch
    # ------------------------------------------------------------------

    def _drain_spatial(self, list_attr: str, accum_attr: str) -> None:
        lst = getattr(self, list_attr)
        if not lst:
            return
        accum = getattr(self, accum_attr)
        for item in lst:
            if accum is None:
                accum = {k: np.zeros_like(v) for k, v in item.items()}
                setattr(self, accum_attr, accum)
            for k in accum:
                accum[k] += item[k]
        lst.clear()

    # ------------------------------------------------------------------
    # 2. plot_samples: subsample newly added entries back to budget
    # ------------------------------------------------------------------

    def _trim_plot_samples(self, prev_len: int) -> None:
        current_len = len(self.plot_samples["y_true"])
        n_new = current_len - prev_len
        if n_new <= 0:
            return

        remaining = self.max_plot_samples - prev_len
        if remaining <= 0:
            for key in self.plot_samples:
                del self.plot_samples[key][prev_len:]
            return

        if n_new <= remaining:
            return  # fits within budget, keep all

        keep = np.sort(np.random.choice(n_new, size=remaining, replace=False))
        for key in self.plot_samples:
            new_slice = [self.plot_samples[key][prev_len + i] for i in keep]
            del self.plot_samples[key][prev_len:]
            self.plot_samples[key].extend(new_slice)

    # ------------------------------------------------------------------
    # Override _process_batch to intercept both bottlenecks
    # ------------------------------------------------------------------

    def _process_batch(self, X, y, mask, vhm0, y_pred, timestamps=None,
                       confidence=None, prior_bias=None, batch_idx=0,
                       timestamps_raw=None):
        prev_plot_len = len(self.plot_samples["y_true"])

        super()._process_batch(
            X, y, mask, vhm0, y_pred,
            timestamps=timestamps,
            confidence=confidence,
            prior_bias=prior_bias,
            batch_idx=batch_idx,
            timestamps_raw=timestamps_raw,
        )

        self._drain_spatial("spatial_errors_model",    "_spatial_model_accum")
        self._drain_spatial("spatial_errors_baseline", "_spatial_baseline_accum")
        self._trim_plot_samples(prev_plot_len)

    # ------------------------------------------------------------------
    # Expose running sums as single-element lists before plotting
    # ------------------------------------------------------------------

    def run_inference(self):
        super().run_inference()
        if self._spatial_model_accum is not None:
            self.spatial_errors_model = [self._spatial_model_accum]
        if self._spatial_baseline_accum is not None:
            self.spatial_errors_baseline = [self._spatial_baseline_accum]


# ---------------------------------------------------------------------------
# CLI — all evaluate_bunet.py args plus --max-plot-samples
# ---------------------------------------------------------------------------

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--max-plot-samples", type=int, default=10_000_000)
    known, _ = pre_parser.parse_known_args()
    max_plot_samples = known.max_plot_samples

    class _Evaluator(LowMemModelEvaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, max_plot_samples=max_plot_samples, **kwargs)

    _base_main(evaluator_class=_Evaluator)


if __name__ == "__main__":
    main()
