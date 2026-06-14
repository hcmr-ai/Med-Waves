#!/usr/bin/env python3
"""Render clean config-driven HTML schematics for TransUNet and MoE TransUNet."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import yaml


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


def base_css() -> str:
    return """
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #475569;
      --blue: #3b82f6;
      --orange: #fdba74;
      --green: #bbf7d0;
      --pink: #f5d0fe;
      --lav: #ddd6fe;
      --io: #bfdbfe;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 32px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #eff6ff 0, transparent 30%),
        radial-gradient(circle at top right, #fdf2f8 0, transparent 24%),
        linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .page {
      max-width: 1800px;
      margin: 0 auto;
      background: rgba(255,255,255,0.7);
      backdrop-filter: blur(6px);
      border: 1px solid rgba(148,163,184,0.25);
      border-radius: 28px;
      padding: 28px 28px 18px;
      box-shadow: var(--shadow);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 56px;
      line-height: 1;
      letter-spacing: -0.03em;
    }
    .subtitle {
      margin: 0 0 26px;
      color: var(--muted);
      font-size: 28px;
    }
    .legend {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin-top: 20px;
      color: var(--muted);
      font-size: 20px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148,163,184,0.35);
      padding: 10px 14px;
      border-radius: 999px;
    }
    .swatch {
      width: 24px;
      height: 24px;
      border-radius: 8px;
      border: 1px solid rgba(71,85,105,0.45);
    }
    .box {
      border: 2px solid rgba(51,65,85,0.78);
      border-radius: 24px;
      padding: 18px 20px;
      text-align: center;
      box-shadow: var(--shadow);
    }
    .box.io { background: var(--io); }
    .box.encoder { background: #ffedd5; }
    .box.feature { background: var(--green); }
    .box.mix { background: #fae8ff; }
    .box.aux { background: var(--lav); }
    .tiny {
      font-size: 18px;
      color: var(--muted);
      margin-top: 6px;
    }
    .footer {
      margin-top: 26px;
      text-align: center;
      color: var(--muted);
      font-size: 22px;
    }
    .arrow {
      color: var(--blue);
      font-size: 34px;
      line-height: 1;
      font-weight: 700;
      text-align: center;
    }
    .down {
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      justify-self: center;
    }
    """


def html_doc(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
  {base_css()}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def render_transunet(config: dict) -> str:
    model = config["model"]
    base = int(model["transunet_base_channels"])
    bottleneck = int(model["transunet_bottleneck_dim"])
    patch = int(model["transunet_patch_size"])
    layers = int(model["transunet_num_layers"])
    heads = int(model["transunet_num_heads"])
    chans = stage_channels(base)
    task_list = ", ".join(tasks(config))

    body = f"""
<div class="page">
  <h1>TransUNet Schematic</h1>
  <p class="subtitle">in_channels={model['in_channels']} | tasks={html.escape(task_list)} | base={base} | bottleneck={bottleneck}</p>

  <div style="display:grid;grid-template-columns:260px 90px 440px 90px 720px 180px;gap:18px;align-items:start;">
    <div class="box io" style="margin-top:180px;">
      <div style="font-size:28px;font-weight:700;">Input tensor</div>
      <div style="font-size:26px;margin-top:8px;">C={model['in_channels']}</div>
      <div style="font-size:26px;margin-top:6px;">64×64 grid</div>
    </div>

    <div class="arrow" style="margin-top:220px;">→</div>

    <div class="box encoder" style="padding:22px;border-style:dashed;">
      <div style="font-size:34px;font-weight:800;margin-bottom:18px;">Transformer Branch</div>
      <div class="box encoder" style="margin-bottom:16px;">
        <div style="font-size:24px;font-weight:700;">Patch embedding</div>
        <div class="tiny">patch size = {patch}</div>
      </div>
      <div class="arrow down">→</div>
      <div class="box encoder" style="margin-top:16px;">
        <div style="font-size:24px;font-weight:700;">Transformer encoder</div>
        <div class="tiny">{layers} layer(s), {heads} head(s)</div>
        <div class="tiny">coordinate positional encoding optional</div>
      </div>
    </div>

    <div class="arrow" style="margin-top:220px;">→</div>

    <div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:22px;align-items:start;">
        {''.join(
            f'''
            <div>
              <div class="box encoder">
                <div style="font-size:24px;font-weight:700;">Encoder {i+1}</div>
                <div style="font-size:28px;margin-top:6px;">{c}</div>
                <div class="tiny">{s}</div>
              </div>
            </div>
            ''' for i, (c, s) in enumerate(zip(chans, ["64×64","32×32","16×16","8×8"], strict=False))
        )}
      </div>
      <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;align-items:center;margin:12px 0 14px;">
        <div></div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div></div>
      </div>
      <div style="display:grid;grid-template-columns:160px 90px 160px 90px repeat(4,1fr);gap:22px;align-items:center;">
        <div class="box mix">
          <div style="font-size:22px;font-weight:700;">Fusion</div>
          <div class="tiny">concat + 1×1 conv</div>
        </div>
        <div class="arrow">→</div>
        <div class="box feature">
          <div style="font-size:24px;font-weight:700;">Bottleneck</div>
          <div style="font-size:28px;margin-top:6px;">{bottleneck}</div>
          <div class="tiny">8×8</div>
        </div>
        <div class="arrow">→</div>
        {''.join(
            f'''
            <div class="box feature">
              <div style="font-size:24px;font-weight:700;">Decoder {i+1}</div>
              <div style="font-size:28px;margin-top:6px;">{c}</div>
              <div class="tiny">{s}</div>
            </div>
            ''' for i, (c, s) in enumerate(zip(chans[::-1], ["8×8","16×16","32×32","64×64"], strict=False))
        )}
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:22px;align-items:center;margin-top:10px;">
        <div class="tiny">skip</div><div class="tiny">skip</div><div class="tiny">skip</div><div class="tiny">skip</div>
      </div>
    </div>

    <div>
      <div class="box mix" style="margin-top:180px;">
        <div style="font-size:24px;font-weight:700;">Task heads</div>
        <div class="tiny">one 1×1 conv per task</div>
      </div>
      <div class="arrow down" style="margin:10px auto;">→</div>
      <div class="box io">
        <div style="font-size:24px;font-weight:700;">Prediction</div>
      </div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background:var(--io)"></span>Input / output</div>
    <div class="legend-item"><span class="swatch" style="background:#ffedd5"></span>Encoder / transformer</div>
    <div class="legend-item"><span class="swatch" style="background:var(--green)"></span>Bottleneck / decoder</div>
    <div class="legend-item"><span class="swatch" style="background:#fae8ff"></span>Fusion / task heads</div>
  </div>

  <div class="footer">Transformer branch + CNN encoder → fused bottleneck → decoder → task heads</div>
</div>
"""
    return html_doc("TransUNet Schematic", body)


def render_moe(config: dict) -> str:
    model = config["model"]
    base = int(model["transunet_base_channels"])
    bottleneck = int(model["transunet_bottleneck_dim"])
    experts = int(model["num_experts"])
    chans = stage_channels(base)
    task_list = ", ".join(tasks(config))
    gate_mode = model.get("gate_input_mode", "features")

    body = f"""
<div class="page">
  <h1>MoE TransUNet Schematic</h1>
  <p class="subtitle">in_channels={model['in_channels']} | tasks={html.escape(task_list)} | experts={experts} | gate_mode={html.escape(gate_mode)}</p>

  <div style="display:grid;grid-template-columns:260px 80px 900px 120px 300px 100px 260px;gap:18px;align-items:start;">
    <div class="box io" style="margin-top:220px;">
      <div style="font-size:28px;font-weight:700;">Input tensor</div>
      <div style="font-size:26px;margin-top:8px;">C={model['in_channels']}</div>
      <div style="font-size:26px;margin-top:6px;">64×64 grid</div>
    </div>

    <div class="arrow" style="margin-top:262px;">→</div>

    <div class="box" style="border-style:dashed;padding:22px;">
      <div style="font-size:42px;font-weight:800;margin-bottom:18px;">Shared TransUNet Backbone</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:22px;">
        {''.join(
            f'''
            <div class="box encoder">
              <div style="font-size:24px;font-weight:700;">Encoder {i+1}</div>
              <div style="font-size:30px;margin-top:6px;">{c}</div>
              <div class="tiny">{s}</div>
            </div>
            ''' for i, (c, s) in enumerate(zip(chans, ["64×64","32×32","16×16","8×8"], strict=False))
        )}
      </div>
      <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;align-items:center;margin:10px 0 16px;">
        <div></div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div class="arrow">→</div><div></div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:22px;align-items:center;">
        <div class="box mix">
          <div style="font-size:22px;font-weight:700;">Fusion</div>
          <div class="tiny">CNN + transformer</div>
        </div>
        <div class="box feature">
          <div style="font-size:24px;font-weight:700;">Bottleneck</div>
          <div style="font-size:30px;margin-top:6px;">{bottleneck}</div>
          <div class="tiny">8×8</div>
        </div>
        <div class="box feature">
          <div style="font-size:24px;font-weight:700;">Decoder 1</div>
          <div style="font-size:30px;margin-top:6px;">{chans[3]}</div>
          <div class="tiny">8×8</div>
        </div>
        <div class="box feature">
          <div style="font-size:24px;font-weight:700;">Decoder 2</div>
          <div style="font-size:30px;margin-top:6px;">{chans[2]}</div>
          <div class="tiny">16×16</div>
        </div>
        <div class="box feature">
          <div style="font-size:24px;font-weight:700;">Shared features</div>
          <div style="font-size:30px;margin-top:6px;">{chans[0]}</div>
          <div class="tiny">64×64</div>
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-rows:180px 130px 130px;align-items:center;">
      <div class="arrow">→</div>
      <div class="arrow">→</div>
      <div class="arrow">→</div>
    </div>

    <div>
      <div class="box encoder" style="margin-top:70px;">
        <div style="font-size:30px;font-weight:800;">Gate head</div>
        <div style="font-size:26px;margin-top:8px;">mode={html.escape(gate_mode)}</div>
        <div class="tiny">K={experts} logits</div>
      </div>
      <div class="box mix" style="margin-top:32px;">
        <div style="font-size:30px;font-weight:800;">Expert heads</div>
        <div style="font-size:24px;margin-top:8px;">{experts} per task</div>
        <div class="tiny">1×1 conv each</div>
      </div>
    </div>

    <div style="display:grid;grid-template-rows:180px 180px 120px;align-items:center;">
      <div class="arrow">→</div>
      <div class="arrow">→</div>
      <div class="arrow down">→</div>
    </div>

    <div>
      <div class="box mix" style="margin-top:70px;">
        <div style="font-size:30px;font-weight:800;">Softmax / temperature</div>
        <div class="tiny">gate weights</div>
      </div>
      <div class="box feature" style="margin-top:32px;">
        <div style="font-size:30px;font-weight:800;">Weighted sum</div>
        <div style="font-size:22px;margin-top:8px;">Σ gate_k · expert_k</div>
      </div>
      <div class="box aux" style="margin-top:32px;">
        <div style="font-size:28px;font-weight:800;">Uncertainty</div>
        <div class="tiny">1 - max gate</div>
      </div>
      <div class="arrow down" style="margin:12px auto;">→</div>
      <div class="box io">
        <div style="font-size:28px;font-weight:800;">Prediction</div>
      </div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background:var(--io)"></span>Input / output</div>
    <div class="legend-item"><span class="swatch" style="background:#ffedd5"></span>Encoder / gate blocks</div>
    <div class="legend-item"><span class="swatch" style="background:var(--green)"></span>Bottleneck / decoder / shared features</div>
    <div class="legend-item"><span class="swatch" style="background:#fae8ff"></span>Experts / mixture / softmax</div>
    <div class="legend-item"><span class="swatch" style="background:var(--lav)"></span>Uncertainty</div>
  </div>

  <div class="footer">Shared backbone → gate branch + expert branch → weighted mixture → prediction</div>
</div>
"""
    return html_doc("MoE TransUNet Schematic", body)


def main():
    parser = argparse.ArgumentParser(description="Render clean HTML schematics for TransUNet and MoE TransUNet.")
    parser.add_argument("--config", default="src/configs/config_dnn.yaml")
    parser.add_argument("--output-dir", default="artifacts/architecture_plots")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trans_path = out_dir / "transunet_schematic.html"
    moe_path = out_dir / "moe_transunet_schematic.html"

    trans_path.write_text(render_transunet(config), encoding="utf-8")
    moe_path.write_text(render_moe(config), encoding="utf-8")

    print(f"Wrote {trans_path}")
    print(f"Wrote {moe_path}")


if __name__ == "__main__":
    main()
