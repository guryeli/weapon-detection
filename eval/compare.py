"""
Correct, unit-consistent model comparison.

Problems in the old `benchmark_comparison.py`:
- YOLO 0-1 and Faster R-CNN 0-100 scales went through the same *100,
  producing nonsense numbers such as ~9500% for Faster R-CNN.
- The Faster R-CNN precision/recall/F1 calculation was wrong.

This script reads the v2 JSON files (output of recompute_benchmarks.py):
- validates units against the `scale: "0-1"` field
- normalizes every metric to a 0-100 percent
- writes a table + JSON summary + matplotlib figures + an HTML report
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_benchmark(path: Path) -> Dict:
    """Read a v2 JSON file and validate the scale."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = json.loads(path.read_text())
    scale = data.get("scale")
    if scale != "0-1":
        raise ValueError(
            f"{path.name}: expected scale='0-1' but got '{scale}'. "
            "Run recompute_benchmarks.py first."
        )
    return data


def to_percent(v: float) -> float:
    return v * 100.0


def summary_row(b: Dict) -> Dict[str, float]:
    """One row for the comparison table (percent / FPS / ms)."""
    s = b["performance_summary"]
    info = b["model_info"]
    return {
        "model": info["model_name"],
        "framework": info["framework"],
        "params_M": info.get("parameters", 0) / 1e6,
        "input_size": info.get("input_size"),
        "device": info.get("device"),
        "mAP50_pct": to_percent(s["mAP50"]),
        "mAP50_95_pct": to_percent(s["mAP50_95"]),
        "precision_pct": to_percent(s["precision"]),
        "recall_pct": to_percent(s["recall"]),
        "f1_pct": to_percent(s["f1_score"]),
        "fps_median": s.get("fps_median", 0.0),
        "median_ms": s.get("median_inference_ms", 0.0),
        "best_thr": b["validation"].get("best_confidence_threshold", 0.0),
    }


def print_table(rows: List[Dict[str, float]]) -> str:
    """Print an ASCII table and return it."""
    cols = [
        ("Model", "model", "{:<24}"),
        ("Framework", "framework", "{:<14}"),
        ("Params(M)", "params_M", "{:>9.2f}"),
        ("Input", "input_size", "{:>5}"),
        ("Device", "device", "{:>6}"),
        ("mAP@50(%)", "mAP50_pct", "{:>9.2f}"),
        ("mAP[.5:.95](%)", "mAP50_95_pct", "{:>14.2f}"),
        ("Prec(%)", "precision_pct", "{:>7.2f}"),
        ("Recall(%)", "recall_pct", "{:>9.2f}"),
        ("F1(%)", "f1_pct", "{:>6.2f}"),
        ("Best.thr", "best_thr", "{:>8.2f}"),
        ("FPS", "fps_median", "{:>6.1f}"),
        ("ms/img", "median_ms", "{:>7.1f}"),
    ]

    lines = []
    header = "  ".join(h for h, _, _ in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        row = "  ".join(f.format(r.get(k, 0)) for _, k, f in cols)
        lines.append(row)
    out = "\n".join(lines)
    print(out)
    return out


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------

def plot_pr_curves(benchmarks: List[Dict], outpath: Path) -> None:
    """Plot Precision-Recall curves from the confidence sweep.
    Order is (R, P); the typical trend as thresholds rise is P up, R down.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for b in benchmarks:
        sw = b["validation"]["sweep"]
        recall = sw["recall"]
        precision = sw["precision"]
        # recall usually falls as the threshold rises; sort by recall for the curve
        pairs = sorted(zip(recall, precision))
        rs = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        ax.plot(rs, ps, marker="o", markersize=3,
                label=b["model_info"]["model_name"])
        # mark the F1-best point
        ax.scatter([b["validation"]["recall_at_best_f1"]],
                   [b["validation"]["precision_at_best_f1"]],
                   s=80, edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–Recall Curve (IoU=0.5, val set)\n"
                 "Point with a black outline: F1-optimal threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_metrics_bar(rows: List[Dict[str, float]], outpath: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = [("mAP@50", "mAP50_pct"),
               ("mAP[.5:.95]", "mAP50_95_pct"),
               ("Precision", "precision_pct"),
               ("Recall", "recall_pct"),
               ("F1", "f1_pct")]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        vals = [r.get(k, 0.0) for _, k in metrics]
        bars = ax.bar(x + offset, vals, width, label=r["model"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in metrics])
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Validation Metric Comparison (percent)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_speed_vs_accuracy(rows: List[Dict[str, float]], outpath: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        ax.scatter(r["fps_median"], r["mAP50_pct"], s=200,
                   label=f"{r['model']} ({r['params_M']:.1f}M)")
        ax.annotate(r["model"],
                    (r["fps_median"], r["mAP50_pct"]),
                    xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel("FPS (median)")
    ax.set_ylabel("mAP@50 (%)")
    ax.set_title("Speed–Accuracy Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_summary_table(rows: List[Dict[str, float]], outpath: Path) -> None:
    """Table figure with matplotlib (suitable for a thesis appendix)."""
    import matplotlib.pyplot as plt
    cols = ["Model", "Framework", "Params(M)", "mAP@50(%)", "mAP[.5:.95](%)",
            "Precision(%)", "Recall(%)", "F1(%)", "FPS", "ms/img"]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["model"], r["framework"], f"{r['params_M']:.2f}",
            f"{r['mAP50_pct']:.2f}", f"{r['mAP50_95_pct']:.2f}",
            f"{r['precision_pct']:.2f}", f"{r['recall_pct']:.2f}",
            f"{r['f1_pct']:.2f}",
            f"{r['fps_median']:.1f}", f"{r['median_ms']:.1f}",
        ])
    fig, ax = plt.subplots(figsize=(13, 1 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
# HTML report
# -------------------------------------------------------------------------

def build_html(rows: List[Dict[str, float]], figures: Dict[str, Path], outpath: Path) -> None:
    def th_td(v):
        return f"<td>{v}</td>" if not isinstance(v, str) or v[0].isdigit() else f"<td>{v}</td>"

    rows_html = ""
    for r in rows:
        rows_html += "<tr>"
        for v in (r["model"], r["framework"], f"{r['params_M']:.2f}",
                  f"{r['mAP50_pct']:.2f}", f"{r['mAP50_95_pct']:.2f}",
                  f"{r['precision_pct']:.2f}", f"{r['recall_pct']:.2f}",
                  f"{r['f1_pct']:.2f}", f"{r['fps_median']:.1f}",
                  f"{r['median_ms']:.1f}", f"{r['best_thr']:.2f}"):
            rows_html += f"<td>{v}</td>"
        rows_html += "</tr>"

    figs_html = ""
    for title, p in figures.items():
        rel = Path("figures") / p.name
        figs_html += f'<h3>{title}</h3><img src="{rel}" style="max-width:100%;" />\n'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Weapon Detection — Comparison (v2)</title>
<style>
body {{ font-family: -apple-system, system-ui, Arial, sans-serif; max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ color: #335; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: center; }}
th {{ background: #eef; }}
tr:nth-child(even) {{ background: #fafafa; }}
.note {{ color: #555; background: #ffd; padding: 10px; border-left: 4px solid #db4; }}
img {{ display: block; margin: 8px 0 18px; border: 1px solid #ddd; }}
</style></head><body>
<h1>Weapon Detection — Model Comparison (v2)</h1>
<p>Generated: {datetime.now().isoformat(timespec='seconds')}</p>
<p class="note"><b>Methodology:</b> Every metric was computed on the same val set (143 images, 219 annotations, 1 class=weapon).
mAP values come from <code>pycocotools</code> COCOeval; precision/recall/F1 are reported at the
point where F1 is maximized, after scanning the confidence threshold from 0.05 to 0.95 at 50 points.
Speed (FPS) is the median of 30 runs after warmup.</p>
<h2>Summary Table</h2>
<table>
<thead><tr>
<th>Model</th><th>Framework</th><th>Params (M)</th>
<th>mAP@50 (%)</th><th>mAP[.5:.95] (%)</th>
<th>Precision (%)</th><th>Recall (%)</th><th>F1 (%)</th>
<th>FPS</th><th>ms/img</th><th>Best thr</th>
</tr></thead>
<tbody>
{rows_html}
</tbody></table>
<h2>Figures</h2>
{figs_html}
</body></html>"""
    outpath.write_text(html)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo",
                    default=str(BENCHMARK_DIR / "yolov8_benchmark_v2.json"))
    ap.add_argument("--frcnn",
                    default=str(BENCHMARK_DIR / "faster_rcnn_benchmark_v2.json"))
    ap.add_argument("--out-json",
                    default=str(BENCHMARK_DIR / "final_benchmark_comparison_v2.json"))
    ap.add_argument("--out-html",
                    default=str(BENCHMARK_DIR / "benchmark_report_v2.html"))
    args = ap.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)
    benchmarks: List[Dict] = []
    for p in (Path(args.yolo), Path(args.frcnn)):
        if p.exists():
            benchmarks.append(load_benchmark(p))
        else:
            print(f"[WARN] Skipped (missing): {p}")

    if not benchmarks:
        print("[ERROR] No benchmark file could be read.")
        return

    rows = [summary_row(b) for b in benchmarks]
    print(f"\n{'='*120}\nComparison Table\n{'='*120}")
    table_str = print_table(rows)

    # Plots
    figures: Dict[str, Path] = {}
    fp = FIGURES_DIR / "pr_curves.png"; plot_pr_curves(benchmarks, fp); figures["Precision–Recall Curve"] = fp
    fp = FIGURES_DIR / "metrics_bar.png"; plot_metrics_bar(rows, fp); figures["Metric Comparison"] = fp
    fp = FIGURES_DIR / "speed_vs_accuracy.png"; plot_speed_vs_accuracy(rows, fp); figures["Speed – Accuracy"] = fp
    fp = FIGURES_DIR / "summary_table.png"; plot_summary_table(rows, fp); figures["Summary Table"] = fp

    # JSON summary
    summary = {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "raw_benchmarks": [{"path": str(p), "scale": b.get("scale")}
                           for b, p in zip(benchmarks,
                                           (Path(args.yolo), Path(args.frcnn))[:len(benchmarks)])],
        "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in figures.items()},
        "ascii_table": table_str,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[OK] JSON: {args.out_json}")

    build_html(rows, figures, Path(args.out_html))
    print(f"[OK] HTML: {args.out_html}")
    print(f"[OK] Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
