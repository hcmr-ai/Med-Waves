#!/usr/bin/env python3
"""Render clean config-driven SVG schematics for TransUNet and MoE TransUNet."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import yaml


class Svg:
    def __init__(self, width: int = 1800, height: int = 1000):
        self.width = width
        self.height = height
        self.parts: list[str] = [
            """
<defs>
  <marker id="arrow-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#1d4ed8" />
  </marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
  </marker>
</defs>
""".strip()
        ]

    def rect(self, x, y, w, h, fill, stroke="#1f2937", stroke_width=2, rx=14, dashed=False):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash} />'
        )

    def line(self, x1, y1, x2, y2, color="#1d4ed8", width=3, marker="arrow-blue", dashed=False):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" marker-end="url(#{marker})"{dash} />'
        )

    def text(self, x, y, value, size=22, weight="normal", fill="#111827", anchor="middle"):
        lines = value.split("\n")
        step = size * 1.25
        start = y - ((len(lines) - 1) * step / 2)
        for idx, line in enumerate(lines):
            yy = start + idx * step
            self.parts.append(
                f'<text x="{x}" y="{yy}" text-anchor="{anchor}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
                f'font-weight="{weight}" fill="{fill}">{html.escape(line)}</text>'
            )

    def save(self, path: Path):
        body = "\n".join(self.parts)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="100%" height="100%" fill="#ffffff" />\n{body}\n</svg>\n'
        )
        path.write_text(svg, encoding="utf-8")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def tasks(config: dict) -> list[str]:
    tasks_config = config["model"].get("tasks_config")
    if tasks_config:
        return [task["name"] for task in tasks_config]
    return list(config["data"].get("target_columns", {"vhm0": "corrected_VHM0"}).keys())


def stage_channels(base: int) -> list[int]:
    return [base, base * 2, base * 4, base * 8]


def draw_title(svg: Svg, title: str, subtitle: str):
    svg.text(60, 45, title, size=34, weight="bold", anchor="start")
    svg.text(60, 82, subtitle, size=18, fill="#475569", anchor="start")


def draw_legend(svg: Svg, x: int, y: int):
    svg.rect(x, y, 280, 210, "#ffffff", stroke="#94a3b8", dashed=True)
    items = [
        ("#bfdbfe", "Input / output"),
        ("#fed7aa", "Encoder / transformer blocks"),
        ("#bbf7d0", "Bottleneck / decoder / features"),
        ("#fce7f3", "Fusion / task heads / mixture"),
    ]
    yy = y + 34
    for color, label in items:
        svg.rect(x + 20, yy - 16, 32, 22, color, stroke="#64748b", stroke_width=1, rx=6)
        svg.text(x + 70, yy, label, size=18, anchor="start")
        yy += 42
    svg.line(x + 22, y + 178, x + 62, y + 178)
    svg.text(x + 78, y + 184, "Main flow / skip flow", size=18, anchor="start")


def draw_encoder_decoder(svg: Svg, base: int, bottleneck: int):
    ch = stage_channels(base)
    enc_x = [460, 620, 780, 940]
    dec_x = [1160, 1320, 1480, 1640]
    y_top = [150, 280, 420, 570]
    sizes = ["64×64", "32×32", "16×16", "8×8"]

    for idx, (x, y, c, size) in enumerate(zip(enc_x, y_top, ch, sizes, strict=False)):
        svg.rect(x, y, 70, 120, "#fed7aa")
        svg.text(x + 35, y - 16, f"{c}", size=20)
        svg.text(x + 35, y + 140, size, size=18)
        if idx < len(enc_x) - 1:
            svg.line(x + 70, y + 60, enc_x[idx + 1], y_top[idx + 1] + 60)

    svg.rect(1070, 590, 90, 90, "#fce7f3")
    svg.text(1115, 578, f"{bottleneck}", size=20)
    svg.text(1115, 708, "bottleneck", size=18)

    svg.line(1010, 630, 1070, 635)

    dec_ch = [ch[3], ch[2], ch[1], ch[0]]
    dec_sizes = ["8×8", "16×16", "32×32", "64×64"]
    dec_y = [570, 420, 280, 150]
    for idx, (x, y, c, size) in enumerate(zip(dec_x, dec_y, dec_ch, dec_sizes, strict=False)):
        svg.rect(x, y, 70, 120, "#bbf7d0")
        svg.text(x + 35, y - 16, f"{c}", size=20)
        svg.text(x + 35, y + 140, size, size=18)
        if idx == 0:
            svg.line(1160, 635, x, y + 60)
        else:
            svg.line(dec_x[idx - 1] + 70, dec_y[idx - 1] + 60, x, y + 60)

    # skip connections
    for ex, ey, dx, dy in zip(enc_x, y_top, dec_x[::-1], dec_y[::-1], strict=False):
        svg.line(ex + 70, ey + 20, dx, dy + 20, color="#60a5fa", width=2, dashed=False)


def render_transunet(config: dict, output_path: Path):
    model = config["model"]
    svg = Svg()
    base = int(model["transunet_base_channels"])
    bottleneck = int(model["transunet_bottleneck_dim"])
    patch = int(model["transunet_patch_size"])
    n_layers = int(model["transunet_num_layers"])
    n_heads = int(model["transunet_num_heads"])
    task_list = tasks(config)

    draw_title(
        svg,
        "TransUNet Schematic",
        f"in_channels={model['in_channels']} | tasks={', '.join(task_list)} | base={base} | bottleneck={bottleneck}",
    )

    svg.rect(90, 130, 170, 120, "#bfdbfe")
    svg.text(175, 190, f"Input tensor\nC={model['in_channels']}\n64×64 grid", size=24)

    svg.rect(110, 330, 260, 290, "#fff7ed", stroke="#0ea5e9", dashed=True)
    svg.text(240, 360, "Transformer Branch", size=26, weight="bold")
    svg.rect(150, 395, 180, 50, "#d9f99d", stroke="#64748b", stroke_width=1)
    svg.text(240, 425, f"Patch embed\npatch={patch}", size=22)
    svg.rect(150, 475, 180, 110, "#fed7aa", stroke="#64748b", stroke_width=1)
    svg.text(240, 530, f"Transformer encoder\nlayers={n_layers}\nheads={n_heads}", size=22)

    svg.line(175, 250, 175, 330)
    svg.line(260, 250, 930, 250, color="#60a5fa", width=2)
    svg.text(610, 232, "CNN encoder path", size=18, fill="#475569")

    draw_encoder_decoder(svg, base, bottleneck)

    svg.line(330, 530, 1070, 630)
    svg.text(690, 558, "transformer features fused with CNN bottleneck", size=18, fill="#475569")

    svg.rect(1710, 130, 90, 120, "#fce7f3")
    svg.text(1755, 190, "Task\nheads", size=24)
    svg.rect(1710, 330, 90, 60, "#bfdbfe")
    svg.text(1755, 365, "Output", size=24)
    svg.line(1710, 210, 1730, 330, marker="arrow-orange", color="#f59e0b")

    draw_legend(svg, 1450, 740)
    svg.save(output_path)


def render_moe(config: dict, output_path: Path):
    model = config["model"]
    svg = Svg(width=1900, height=1040)
    base = int(model["transunet_base_channels"])
    bottleneck = int(model["transunet_bottleneck_dim"])
    n_experts = int(model["num_experts"])
    gate_mode = model.get("gate_input_mode", "features")
    task_list = tasks(config)

    draw_title(
        svg,
        "MoE TransUNet Schematic",
        f"in_channels={model['in_channels']} | tasks={', '.join(task_list)} | experts={n_experts} | gate_mode={gate_mode}",
    )

    svg.rect(80, 160, 180, 130, "#bfdbfe")
    svg.text(170, 225, f"Input tensor\nC={model['in_channels']}\n64×64 grid", size=24)

    svg.rect(360, 120, 860, 650, "#f8fafc", stroke="#94a3b8", dashed=True)
    svg.text(790, 155, "Shared TransUNet Backbone", size=30, weight="bold")
    draw_encoder_decoder(svg, base, bottleneck)

    svg.rect(1280, 180, 220, 120, "#fed7aa")
    svg.text(1390, 240, f"Gate head\nmode={gate_mode}\nK={n_experts} logits", size=24)
    svg.rect(1560, 180, 220, 120, "#fce7f3")
    svg.text(1670, 240, "Softmax /\ntemperature\nGate weights", size=24)

    svg.rect(1280, 390, 220, 210, "#fce7f3")
    svg.text(1390, 495, f"Expert heads\n{n_experts} per task\n1×1 conv each", size=24)
    svg.rect(1560, 420, 220, 150, "#bbf7d0")
    svg.text(1670, 495, "Weighted sum\nΣ gate_k · expert_k", size=24)

    svg.rect(1560, 640, 220, 100, "#ddd6fe")
    svg.text(1670, 690, "Uncertainty\n1 - max gate", size=24)
    svg.rect(1560, 790, 220, 100, "#bfdbfe")
    svg.text(1670, 840, "Prediction", size=24)

    svg.line(1220, 300, 1280, 240)
    svg.line(1220, 600, 1280, 495)
    svg.line(1500, 240, 1560, 240)
    svg.line(1500, 495, 1560, 495)
    svg.line(1670, 300, 1670, 420)
    svg.line(1670, 570, 1670, 640)
    svg.line(1670, 740, 1670, 790, marker="arrow-orange", color="#f59e0b")

    svg.text(1260, 1000, "Shared backbone -> gate branch + expert branch -> weighted mixture -> prediction", size=18, fill="#475569")
    draw_legend(svg, 90, 790)
    svg.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render clean SVG schematics for TransUNet and MoE TransUNet.")
    parser.add_argument("--config", default="src/configs/config_dnn.yaml")
    parser.add_argument("--output-dir", default="artifacts/architecture_plots")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trans_path = output_dir / "transunet_schematic.svg"
    moe_path = output_dir / "moe_transunet_schematic.svg"

    render_transunet(config, trans_path)
    render_moe(config, moe_path)

    print(f"Wrote {trans_path}")
    print(f"Wrote {moe_path}")


if __name__ == "__main__":
    main()
