#!/usr/bin/env python3
"""Render TransUNet and MoE TransUNet graphs with torchview and torchviz.

Requirements in the runtime environment:
- graphviz system package with `dot` on PATH
- python packages: torchview, graphviz
- optional python package: torchviz (for autograd execution graph)

Example:
    poetry run python scripts/plot_transunet_architectures.py \
      --config src/configs/config_dnn.yaml \
      --output-dir artifacts/architecture_plots \
      --height 64 --width 64 \
      --include-autograd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.model_factory import create_model


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_tasks(config: dict[str, Any]) -> list[str]:
    tasks_config = config["model"].get("tasks_config")
    if tasks_config:
        return [task["name"] for task in tasks_config]
    return list(config["data"].get("target_columns", {"vhm0": "corrected_VHM0"}).keys())


def build_model(config: dict[str, Any], model_type: str) -> torch.nn.Module:
    model_cfg = config["model"]
    tasks = resolve_tasks(config)
    return create_model(
        model_type=model_type,
        in_channels=model_cfg["in_channels"],
        filters=model_cfg.get("filters"),
        dropout=model_cfg.get("dropout", 0.0),
        add_vhm0_residual=model_cfg.get("add_vhm0_residual", False),
        vhm0_channel_index=model_cfg.get("vhm0_channel_index", 0),
        upsample_mode=model_cfg.get("upsample_mode", "nearest"),
        use_mdn=model_cfg.get("use_mdn", False),
        auxiliary_tasks=tasks,
        transunet_base_channels=model_cfg.get("transunet_base_channels", 32),
        transunet_bottleneck_dim=model_cfg.get("transunet_bottleneck_dim", 512),
        transunet_patch_size=model_cfg.get("transunet_patch_size", 8),
        transunet_num_layers=model_cfg.get("transunet_num_layers", 4),
        transunet_num_heads=model_cfg.get("transunet_num_heads", 8),
        transformer_use_coord_pos_enc=model_cfg.get("transformer_use_coord_pos_enc", True),
        transformer_sea_mask_channel_index=model_cfg.get("transformer_sea_mask_channel_index"),
        num_experts=model_cfg.get("num_experts", 3),
        gate_temperature=model_cfg.get("gate_temperature", 1.0),
        gate_input_mode=model_cfg.get("gate_input_mode", "features"),
        gate_input_channels=model_cfg.get("gate_input_channels"),
        expert_dropout=model_cfg.get("expert_dropout", 0.0),
        transformer_dropout=model_cfg.get("transformer_dropout", 0.0),
        return_gate_maps=model_cfg.get("return_gate_maps", True),
    )


def extract_primary_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "prediction" in output:
            return extract_primary_tensor(output["prediction"])
        first_key = next(iter(output))
        return extract_primary_tensor(output[first_key])
    if isinstance(output, (list, tuple)):
        return extract_primary_tensor(output[0])
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def render_torchview_graph(
    model: torch.nn.Module,
    output_dir: Path,
    stem: str,
    input_shape: tuple[int, int, int, int],
    depth: int,
    expand_nested: bool,
) -> Path:
    from torchview import draw_graph

    def _draw(device: str):
        kwargs: dict[str, Any] = {
            "model": model,
            "graph_name": stem,
            "depth": depth,
            "device": device,
            "expand_nested": expand_nested,
            "hide_inner_tensors": False,
            "hide_module_functions": False,
            "show_shapes": True,
            "save_graph": False,
        }
        if device == "meta":
            kwargs["input_size"] = input_shape
        else:
            kwargs["input_data"] = torch.randn(*input_shape, device=device)
        return draw_graph(**kwargs)

    try:
        graph = _draw("meta")
    except NotImplementedError as exc:
        if "meta tensor" not in str(exc).lower():
            raise
        model = model.to("cpu").eval()
        graph = _draw("cpu")

    visual_graph = getattr(graph, "visual_graph", None)
    if visual_graph is None:
        raise RuntimeError("torchview did not return a visual_graph object.")

    visual_graph.format = "svg"
    rendered = visual_graph.render(
        filename=stem,
        directory=str(output_dir),
        cleanup=True,
    )
    out_path = Path(rendered)
    if not out_path.exists():
        raise FileNotFoundError(f"torchview render did not produce expected file: {out_path}")
    return out_path


def render_torchviz_graph(
    model: torch.nn.Module,
    output_dir: Path,
    stem: str,
    input_shape: tuple[int, int, int, int],
    device: str = "cpu",
) -> Path:
    from torchviz import make_dot

    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    y = model(x)
    primary = extract_primary_tensor(y)
    graph = make_dot(
        primary.mean(),
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    graph.format = "svg"
    rendered = graph.render(filename=stem, directory=str(output_dir), cleanup=True)
    return Path(rendered)


def assert_graphviz_available() -> None:
    import shutil

    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz `dot` is not available on PATH. Install the system Graphviz package first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TransUNet and MoE TransUNet architectures.")
    parser.add_argument("--config", default="src/configs/config_dnn.yaml")
    parser.add_argument("--output-dir", default="artifacts/architecture_plots")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--expand-nested", action="store_true")
    parser.add_argument("--include-autograd", action="store_true")
    args = parser.parse_args()

    assert_graphviz_available()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_channels = int(config["model"]["in_channels"])
    input_shape = (args.batch_size, in_channels, args.height, args.width)

    for model_type in ("transunet", "moe_transunet"):
        torchview_path = render_torchview_graph(
            model=build_model(config, model_type),
            output_dir=output_dir,
            stem=f"{model_type}_torchview",
            input_shape=input_shape,
            depth=args.depth,
            expand_nested=args.expand_nested,
        )
        print(f"Wrote {torchview_path}")

        if args.include_autograd:
            torchviz_path = render_torchviz_graph(
                model=build_model(config, model_type),
                output_dir=output_dir,
                stem=f"{model_type}_torchviz",
                input_shape=input_shape,
            )
            print(f"Wrote {torchviz_path}")


if __name__ == "__main__":
    main()
