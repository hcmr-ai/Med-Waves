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


class TransUNetBackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(x)


class MoEBackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.backbone.extract_features(x)


class MoEGateWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.backbone.extract_features(x)
        if self.model.gate_input_mode == "input_channels":
            gate_input = x[:, self.model.gate_input_channels]
            gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=0.0, neginf=0.0)
            if gate_input.shape[-2:] != features.shape[-2:]:
                gate_input = torch.nn.functional.interpolate(
                    gate_input,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return self.model.gate_head(gate_input)
        return self.model.gate_head(features)


class MoEExpertWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, task_name: str, expert_index: int = 0):
        super().__init__()
        self.model = model
        self.task_name = task_name
        self.expert_index = expert_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.backbone.extract_features(x)
        head = self.model.expert_heads[self.task_name][self.expert_index]
        return head(features)


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
    hide_inner_tensors: bool,
    hide_module_functions: bool,
    roll: bool,
) -> Path:
    from torchview import draw_graph

    def _draw(device: str):
        kwargs: dict[str, Any] = {
            "model": model,
            "graph_name": stem,
            "depth": depth,
            "device": device,
            "expand_nested": expand_nested,
            "hide_inner_tensors": hide_inner_tensors,
            "hide_module_functions": hide_module_functions,
            "show_shapes": True,
            "save_graph": False,
            "roll": roll,
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


def render_torchview_suite(
    config: dict[str, Any],
    output_dir: Path,
    input_shape: tuple[int, int, int, int],
    depth: int,
    expand_nested: bool,
    simple: bool,
) -> list[Path]:
    paths: list[Path] = []
    hide_inner_tensors = simple
    hide_module_functions = simple
    roll = simple

    transunet = build_model(config, "transunet")
    paths.append(
        render_torchview_graph(
            model=TransUNetBackboneWrapper(transunet) if simple else transunet,
            output_dir=output_dir,
            stem="transunet_backbone_torchview" if simple else "transunet_torchview",
            input_shape=input_shape,
            depth=4 if simple else depth,
            expand_nested=expand_nested,
            hide_inner_tensors=hide_inner_tensors,
            hide_module_functions=hide_module_functions,
            roll=roll,
        )
    )
    if simple:
        paths.append(
            render_torchview_graph(
                model=transunet,
                output_dir=output_dir,
                stem="transunet_full_torchview",
                input_shape=input_shape,
                depth=3,
                expand_nested=False,
                hide_inner_tensors=True,
                hide_module_functions=True,
                roll=True,
            )
        )

    moe = build_model(config, "moe_transunet")
    tasks = resolve_tasks(config)
    paths.append(
        render_torchview_graph(
            model=MoEBackboneWrapper(moe) if simple else moe,
            output_dir=output_dir,
            stem="moe_backbone_torchview" if simple else "moe_transunet_torchview",
            input_shape=input_shape,
            depth=4 if simple else depth,
            expand_nested=expand_nested,
            hide_inner_tensors=hide_inner_tensors,
            hide_module_functions=hide_module_functions,
            roll=roll,
        )
    )
    if simple:
        paths.append(
            render_torchview_graph(
                model=MoEGateWrapper(moe),
                output_dir=output_dir,
                stem="moe_gate_torchview",
                input_shape=input_shape,
                depth=3,
                expand_nested=False,
                hide_inner_tensors=True,
                hide_module_functions=True,
                roll=True,
            )
        )
        paths.append(
            render_torchview_graph(
                model=MoEExpertWrapper(moe, task_name=tasks[0], expert_index=0),
                output_dir=output_dir,
                stem=f"moe_expert0_{tasks[0]}_torchview",
                input_shape=input_shape,
                depth=3,
                expand_nested=False,
                hide_inner_tensors=True,
                hide_module_functions=True,
                roll=True,
            )
        )
        paths.append(
            render_torchview_graph(
                model=moe,
                output_dir=output_dir,
                stem="moe_full_torchview",
                input_shape=input_shape,
                depth=3,
                expand_nested=False,
                hide_inner_tensors=True,
                hide_module_functions=True,
                roll=True,
            )
        )
    return paths


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
    parser.add_argument("--simple", action="store_true")
    args = parser.parse_args()

    assert_graphviz_available()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_channels = int(config["model"]["in_channels"])
    input_shape = (args.batch_size, in_channels, args.height, args.width)

    torchview_paths = render_torchview_suite(
        config=config,
        output_dir=output_dir,
        input_shape=input_shape,
        depth=args.depth,
        expand_nested=args.expand_nested,
        simple=args.simple,
    )
    for torchview_path in torchview_paths:
        print(f"Wrote {torchview_path}")

    if args.include_autograd:
        for model_type in ("transunet", "moe_transunet"):
            torchviz_path = render_torchviz_graph(
                model=build_model(config, model_type),
                output_dir=output_dir,
                stem=f"{model_type}_torchviz",
                input_shape=input_shape,
            )
            print(f"Wrote {torchviz_path}")


if __name__ == "__main__":
    main()
