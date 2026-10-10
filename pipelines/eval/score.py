"""
Score grounding predictions against the package ground truth (stdlib only).

  python score.py --pred-dir OUT_DIR --pkg PKG [--split person_test]

Per image: parse the answer (fenced or bare JSON list of {"bbox_2d": [x1,y1,x2,y2], "label": ...});
coordinates are 0-1000 relative and are mapped to PIXELS before IoU; predictions are matched to GT
greedily by IoU. A parse failure counts as a failure (all GT of that image are misses), it is never
dropped. Any box on a negative image is a false positive; boxes with the wrong label are false
positives. An answer in plain text without any box (the base model answers "There is no object ..."
on images without the category) is an EMPTY prediction, counted separately as text_no_boxes -- it
scores exactly like a parse failure (no TP, no FP, every GT box missed) but is not a format error.
Reports P/R/F1 at IoU 0.5 and 0.75, parse-failure / text-without-boxes / truncation / label-mismatch /
coords>1000 rates, and a per-image bootstrap 95% CI for F1@0.5.
"""

import argparse
import json
import random
import re
from pathlib import Path

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
THRESHOLDS = (0.5, 0.75)


def parse_args():
    parser = argparse.ArgumentParser(description="Score grounding predictions")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--split", default="person_test")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def parse(text):
    """List of (label, [x1,y1,x2,y2]) or None when the output is not a valid answer."""
    match = FENCE_RE.search(text)
    body = (match.group(1) if match else text).strip()
    try:
        items = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    boxes = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("bbox_2d"), list) or len(item["bbox_2d"]) != 4:
            return None
        try:
            coords = [float(v) for v in item["bbox_2d"]]
        except (TypeError, ValueError):
            return None
        boxes.append((str(item.get("label", "")), coords))
    return boxes


def to_pixels(box, width, height):
    x1, y1, x2, y2 = (min(1000.0, max(0.0, v)) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_counts(preds, gts, threshold):
    """Greedy one-to-one matching by IoU. Returns (tp, fp, fn)."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    preds = {}
    for path in sorted(pred_dir.glob("preds.rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                preds[row["id"]] = row
    info_path = pred_dir / "predict_info.json"
    if info_path.exists():
        expected = json.loads(info_path.read_text())["n_samples"]
        if len(preds) != expected:
            raise SystemExit(f"{pred_dir}: {len(preds)} predictions but {expected} samples were requested "
                             "(a rank crashed or the run was interrupted) -- not scoring a partial result")
    flat = {json.loads(line)["id"]: json.loads(line)
            for line in (Path(args.pkg) / "records" / f"{args.split}.flat.jsonl").read_text().splitlines() if line.strip()}

    per_image, counters = [], {"parse_fail": 0, "text_no_boxes": 0, "truncated": 0, "label_mismatch": 0,
                               "coords_over_1000": 0, "negatives": 0, "negatives_with_boxes": 0}
    for sample_id, row in sorted(preds.items()):
        gt = flat[sample_id]
        gt_boxes = gt["boxes_pixel"]
        parsed = parse(row["text"])
        counters["truncated"] += int(row.get("truncated", False))
        if gt["is_negative"]:
            counters["negatives"] += 1
        if parsed is None and "[" not in row["text"] and "bbox_2d" not in row["text"]:
            counters["text_no_boxes"] += 1
            parsed = []
        if parsed is None:
            counters["parse_fail"] += 1
            per_image.append({t: (0, 0, len(gt_boxes)) for t in THRESHOLDS})
            continue
        boxes = [(label, to_pixels(coords, gt["width"], gt["height"])) for label, coords in parsed]
        counters["coords_over_1000"] += int(any(v > 1000 for _, coords in parsed for v in coords))
        counters["label_mismatch"] += int(any(label != gt["label"] for label, _ in parsed))
        if gt["is_negative"] and boxes:
            counters["negatives_with_boxes"] += 1
        right = [b for label, b in boxes if label == gt["label"]]
        wrong = len(boxes) - len(right)
        entry = {}
        for t in THRESHOLDS:
            tp, fp, fn = match_counts(right, gt_boxes, t)
            entry[t] = (tp, fp + wrong, fn)
        per_image.append(entry)

    metrics = {"n_images": len(per_image), "n_gt_boxes": sum(e[0.5][0] + e[0.5][2] for e in per_image)}
    for t in THRESHOLDS:
        tp = sum(e[t][0] for e in per_image)
        fp = sum(e[t][1] for e in per_image)
        fn = sum(e[t][2] for e in per_image)
        p, r, f1 = prf(tp, fp, fn)
        metrics[f"iou{t}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
                              "recall": round(r, 4), "f1": round(f1, 4)}
    rng = random.Random(0)
    boot = []
    for _ in range(args.bootstrap):
        draw = [per_image[rng.randrange(len(per_image))] for _ in per_image]
        boot.append(prf(*(sum(e[0.5][k] for e in draw) for k in range(3)))[2])
    boot.sort()
    metrics["f1_iou0.5_ci95"] = [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot)) - 1], 4)]
    n = max(1, len(per_image))
    metrics["rates"] = {k: round(v / n, 4) for k, v in counters.items() if k not in ("negatives", "negatives_with_boxes")}
    metrics["negatives_with_boxes"] = f"{counters['negatives_with_boxes']}/{counters['negatives']}"
    (pred_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    m = metrics["iou0.5"]
    print(f"[score] {pred_dir}: n={metrics['n_images']} F1@0.5={m['f1']:.4f} {metrics['f1_iou0.5_ci95']} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1@0.75={metrics['iou0.75']['f1']:.4f} "
          f"parse_fail={metrics['rates']['parse_fail']:.3f} text_no_boxes={metrics['rates']['text_no_boxes']:.3f} "
          f"truncated={metrics['rates']['truncated']:.3f} neg_with_boxes={metrics['negatives_with_boxes']}")


if __name__ == "__main__":
    main()
