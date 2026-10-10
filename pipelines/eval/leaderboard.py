"""
Collect every run_summary.json and eval metrics under runs/ into runs/leaderboard.md (stdlib only).

Accuracy on n100/n50 subsets is a parity check (F1 +-~5 at 95%); only the full test set with the
bootstrap CI supports a ranking. 6000b 32B rows are sharded (Unsloth: 4-bit QLoRA) -- their speed is
not comparable with H200 replicate rows.
"""

import json
import sys
from pathlib import Path


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(runs):
    runs = Path(runs)
    lines = ["| run | fw | model | strategy | GPUs | batch | steps | rc | load s | step s | samples/s | tokens/s | "
             "peak GPU GiB | host GiB | loss first->last | vision grad | eval | F1@0.5 [CI] | P | R | F1@0.75 | parse fail | "
             "no-box text | truncated |",
             "|" + "---|" * 24]
    for summary_path in sorted(runs.glob("**/run_summary.json")):
        s = json.loads(summary_path.read_text())
        run_dir = summary_path.parent
        evals = sorted(run_dir.glob("eval/*/metrics.json")) or [None]
        for metrics_path in evals:
            m = json.loads(metrics_path.read_text()) if metrics_path else {}
            iou5, iou75 = m.get("iou0.5", {}), m.get("iou0.75", {})
            lines.append("| " + " | ".join([
                s["run"], fmt(s.get("framework")), fmt(s.get("model")), fmt(s.get("strategy")), fmt(s.get("ngpu")),
                fmt(s.get("global_bs")), fmt(s.get("steps_logged")), fmt(s.get("rc")), fmt(s.get("load_s"), 0), fmt(s.get("step_time_s")),
                fmt(s.get("samples_per_s")), fmt(s.get("tokens_per_s"), 0), fmt(s.get("max_peak_mem_gib"), 1),
                fmt(s.get("peak_host_used_gib"), 1),
                f"{fmt(s.get('first_loss'))}->{fmt(s.get('last_loss'))}",
                fmt(s.get("vision_output_requires_grad")),
                metrics_path.parent.name if metrics_path else "-",
                f"{fmt(iou5.get('f1'))} {m.get('f1_iou0.5_ci95', '')}" if iou5 else "-",
                fmt(iou5.get("precision")), fmt(iou5.get("recall")), fmt(iou75.get("f1")),
                fmt(m.get("rates", {}).get("parse_fail")), fmt(m.get("rates", {}).get("text_no_boxes")),
                fmt(m.get("rates", {}).get("truncated")),
            ]) + " |")
    for metrics_path in sorted(runs.glob("baseline/*/eval/*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        iou5, iou75 = m["iou0.5"], m["iou0.75"]
        model = metrics_path.parents[2].name
        lines.append(f"| baseline-{model} | zero-shot | {model} | - | - | - | - | - | - | - | - | - | - | - | - | - | "
                     f"{metrics_path.parent.name} | {fmt(iou5['f1'])} {m['f1_iou0.5_ci95']} | {fmt(iou5['precision'])} | "
                     f"{fmt(iou5['recall'])} | {fmt(iou75['f1'])} | {fmt(m['rates']['parse_fail'])} | "
                     f"{fmt(m['rates'].get('text_no_boxes'))} | {fmt(m['rates'].get('truncated'))} |")
    note = ("\nNotes: n100/n50 accuracy is a parity check only (F1 +-~5 at 95%); rank frameworks on the full test "
            "set with CIs. Sharded 6000b 32B rows (Unsloth: 4-bit QLoRA, reference only) are not speed-comparable "
            "with H200 replicate rows. 'vision grad' must be False (no backward through the frozen ViT). "
            "'parse fail' = malformed or truncated JSON (truncation at max_new_tokens is usually a repetition loop on a "
            "crowded image); 'no-box text' = a plain-text answer without boxes (the base model's way of saying the "
            "category is absent) -- scored as an empty prediction.\n")
    (runs / "leaderboard.md").write_text("# Leaderboard\n\n" + "\n".join(lines) + "\n" + note)
    print(f"leaderboard -> {runs / 'leaderboard.md'} ({len(lines) - 2} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
