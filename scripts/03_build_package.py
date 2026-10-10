"""
Step 03: build the SFT data package (COCO person [+ Compass rock]) for Qwen3-VL.

Per image: PIL-measured raw size must equal the annotation size; images with EXIF
orientation != 1, decode errors, or any box degenerate after 0-1000 quantization are
dropped (counted in meta.json). Boxes are always relative to the raw stored pixels.

Compass (optional, --compass-dir): every category whose name contains "rock" -> "rock"
(anything else fails); images without boxes become negatives; Roboflow augmented copies
(<stem>_<ext>.rf.<hash>) are grouped by original stem -- a stem in two splits fails,
duplicates within a split keep the first. Roboflow train/valid/test is kept as train/val/test.
Without --compass-dir the package is person-only.

Package (all image paths relative to the package root):
  images/coco/{train2017,val2017}/  [images/rock_compass/{train,valid,test}/]
  train.jsonl  = person_train [+ rock_train x --rock-repeat] (seeded shuffle)
  val.jsonl    = person_val [+ rock_val] (seeded shuffle)
  {person,rock}_{train,val,test}.jsonl   per-source rows (ms-swift messages+images format)
  records/<name>.flat.jsonl              flat schema, same row order as <name>.jsonl
  records/{train,val}.ids.txt            source id per row of train.jsonl / val.jsonl
  meta.json  README.md
Training pipelines for ms-swift / TRL / NeMo AutoModel / Axolotl / Unsloth live in pipelines/.

Usage:
    python scripts/03_build_package.py [--compass-dir data/raw/compass/v1] [--force]
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROBOFLOW_STEM_RE = re.compile(r"^(?P<stem>.+)_(?P<ext>[A-Za-z0-9]+)\.rf\.[0-9a-fA-F]+$")
ROCK_SPLITS = {"train": "train", "valid": "val", "test": "test"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build person[/rock] SFT data package")
    parser.add_argument("--coco-dir", type=str, default=str(C.RAW_DIR / "coco"))
    parser.add_argument("--compass-dir", type=str, default=None,
                        help="Compass rock COCO export (omit for a person-only package)")
    parser.add_argument("--out", type=str, default=None,
                        help="default: data/package/vlm_sft_<classes>_<YYMMDD>")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--rock-repeat", type=int, default=2)
    parser.add_argument("--exclude-ids", type=str, default=None,
                        help="text file of sample ids to drop (e.g. rejected in the preview gallery)")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_image(path, width, height):
    """None if usable, else a drop reason."""
    try:
        with Image.open(path) as image:
            orientation = C.exif_orientation(image)
            size = image.size
            image.load()
    except Exception as exc:
        return f"decode_error:{type(exc).__name__}"
    if orientation not in (None, 1):
        return "exif_orientation"
    if size != (width, height):
        return "size_mismatch"
    return None


def quantize_all(boxes_xywh, width, height):
    """Sorted (pixel, quantized) pairs, or None if any box degenerates."""
    try:
        pairs = [C.quantize_box(C.xywh_to_xyxy(b), width, height) for b in boxes_xywh]
    except ValueError:
        return None
    pairs.sort(key=lambda p: tuple(p[1]))
    return pairs


def make_record(sample_id, label, image_rel, width, height, pairs, split, source):
    boxes_1000 = [q for _, q in pairs]
    return {
        "id": sample_id,
        "label": label,
        "image": image_rel,
        "width": width, "height": height,
        "is_negative": not pairs,
        "boxes_pixel": [[round(v, 2) for v in p] for p, _ in pairs],
        "boxes_1000": boxes_1000,
        "target": C.render_answer(label, boxes_1000),
        "split": split,
        "source": source,
    }


def build_person(coco_dir, out_root, drops, excluded_ids):
    selection = json.loads((coco_dir / "selection.json").read_text())
    records = {}
    for split, block in selection["candidates"].items():
        coco_split = block["coco_split"]
        rows = []
        for kind in ("pos", "neg"):
            target = selection["targets"][split][kind]
            taken = 0
            for entry in block[kind]:
                if taken == target:
                    break
                sample_id = f"coco_{coco_split}_{entry['id']:012d}"
                if sample_id in excluded_ids:
                    drops["person"]["excluded_by_list"] += 1
                    continue
                src = coco_dir / coco_split / entry["file_name"]
                reason = check_image(src, entry["width"], entry["height"])
                pairs = quantize_all(entry["boxes_xywh"], entry["width"], entry["height"])
                if reason is None and pairs is None:
                    reason = "degenerate_box"
                if reason:
                    drops["person"][reason] += 1
                    continue
                image_rel = f"images/coco/{coco_split}/{entry['file_name']}"
                dest = out_root / image_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                rows.append(make_record(
                    sample_id, "person", image_rel, entry["width"], entry["height"], pairs, split,
                    {"dataset": "coco2017", "coco_split": coco_split, "image_id": entry["id"],
                     "ann_ids": entry["ann_ids"]}))
                taken += 1
            if taken < target:
                sys.exit(f"person {split}/{kind}: only {taken}/{target} usable -- raise --extra-frac in step 01")
        records[split] = rows
    return records, selection


def original_stem(file_name):
    stem = Path(file_name).stem
    match = ROBOFLOW_STEM_RE.match(stem)
    return match.group("stem") if match else stem


def build_rock(compass_dir, out_root, drops, excluded_ids):
    stem_splits = defaultdict(set)
    loaded = {}
    for rf_split in ROCK_SPLITS:
        ann_path = compass_dir / rf_split / "_annotations.coco.json"
        if not ann_path.exists():
            sys.exit(f"missing {ann_path}")
        payload = json.loads(ann_path.read_text())
        bad = [c["name"] for c in payload["categories"] if "rock" not in c["name"].lower()]
        if bad:
            sys.exit(f"{ann_path}: unexpected non-rock categories {bad}")
        loaded[rf_split] = payload
        for image in payload["images"]:
            stem_splits[original_stem(image["file_name"])].add(rf_split)
    straddling = {s: sorted(v) for s, v in stem_splits.items() if len(v) > 1}
    if straddling:
        sys.exit(f"{len(straddling)} original images straddle splits, e.g. {list(straddling.items())[:5]}")

    records, dup_counts = {}, {}
    for rf_split, split in ROCK_SPLITS.items():
        payload = loaded[rf_split]
        boxes_by_image = defaultdict(list)
        for ann in payload["annotations"]:
            boxes_by_image[ann["image_id"]].append(ann["bbox"])
        seen_stems, rows, dups = set(), [], 0
        for image in sorted(payload["images"], key=lambda i: i["file_name"]):
            stem = original_stem(image["file_name"])
            if stem in seen_stems:
                dups += 1
                continue
            seen_stems.add(stem)
            sample_id = f"compass_{split}_{re.sub(r'[^A-Za-z0-9._-]+', '_', stem)}"
            if sample_id in excluded_ids:
                drops["rock"]["excluded_by_list"] += 1
                continue
            src = compass_dir / rf_split / image["file_name"]
            reason = check_image(src, image["width"], image["height"])
            pairs = quantize_all(boxes_by_image[image["id"]], image["width"], image["height"])
            if reason is None and pairs is None:
                reason = "degenerate_box"
            if reason:
                drops["rock"][reason] += 1
                continue
            image_rel = f"images/rock_compass/{rf_split}/{image['file_name']}"
            dest = out_root / image_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            rows.append(make_record(
                sample_id, "rock", image_rel, image["width"], image["height"], pairs, split,
                {"dataset": "compass-dhncp/rock-detection-9kgmi", "roboflow_split": rf_split,
                 "file_name": image["file_name"], "original_stem": stem}))
        if len({r["id"] for r in rows}) != len(rows):
            sys.exit(f"rock {split}: duplicate sample ids after sanitization")
        records[split] = rows
        dup_counts[split] = dups
    return records, dup_counts


def split_stats(rows):
    boxes = [len(r["boxes_1000"]) for r in rows]
    return {"samples": len(rows), "pos": sum(1 for r in rows if not r["is_negative"]),
            "neg": sum(1 for r in rows if r["is_negative"]), "boxes": sum(boxes),
            "max_boxes_per_image": max(boxes, default=0)}


def main():
    args = parse_args()
    coco_dir = Path(args.coco_dir).resolve()
    compass_dir = Path(args.compass_dir).resolve() if args.compass_dir else None
    classes = ["person"] + (["rock"] if compass_dir else [])
    out_root = Path(args.out or C.PACKAGE_DIR / f"vlm_sft_{'_'.join(classes)}_{datetime.now():%y%m%d}").resolve()
    C.assert_not_dropbox(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        if not args.force:
            sys.exit(f"{out_root} exists and is not empty; use --force to rebuild")
        shutil.rmtree(out_root)
    (out_root / "records").mkdir(parents=True)

    excluded_ids = set()
    if args.exclude_ids:
        excluded_ids = {line.strip() for line in Path(args.exclude_ids).read_text().splitlines()
                        if line.strip()}

    drops = {"person": Counter(), "rock": Counter()}
    person, selection = build_person(coco_dir, out_root, drops, excluded_ids)
    rock, rock_dups = build_rock(compass_dir, out_root, drops, excluded_ids) if compass_dir else ({}, {})

    per_source = {}
    for source, records in (("person", person), ("rock", rock)):
        for split, rows in records.items():
            name = f"{source}_{split}"
            per_source[name] = rows
            C.write_jsonl(out_root / "records" / f"{name}.flat.jsonl", rows)
            C.write_jsonl(out_root / f"{name}.jsonl",
                          [C.swift_row(r["label"], r["image"], r["boxes_1000"]) for r in rows])

    rng = random.Random(args.seed)
    mixes = {
        "train": person["train"] + rock.get("train", []) * args.rock_repeat,
        "val": person["val"] + rock.get("val", []),
    }
    for name, rows in mixes.items():
        rows = list(rows)
        rng.shuffle(rows)
        C.write_jsonl(out_root / f"{name}.jsonl",
                      [C.swift_row(r["label"], r["image"], r["boxes_1000"]) for r in rows])
        (out_root / "records" / f"{name}.ids.txt").write_text("\n".join(r["id"] for r in rows) + "\n")

    compass_version = None
    if compass_dir and (compass_dir / "version_info.json").exists():
        compass_version = json.loads((compass_dir / "version_info.json").read_text())
    stats = {name: split_stats(rows) for name, rows in per_source.items()}
    stats["train_mix"] = {"samples": len(mixes["train"]), "person": len(person["train"]),
                          "rock": len(rock.get("train", [])) * args.rock_repeat}
    stats["val_mix"] = {"samples": len(mixes["val"]), "person": len(person["val"]),
                        "rock": len(rock.get("val", []))}
    meta = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": classes,
        "seed": args.seed,
        "rock_repeat": args.rock_repeat if compass_dir else 0,
        "prompt_template": C.PROMPT_TEMPLATE,
        "answer_format": "Qwen3-VL cookbook fenced JSON list of {bbox_2d:[x1,y1,x2,y2], label}, "
                         "coords 0-1000 relative; negative = ```json\\n[]\\n```",
        "quantization": C.QUANT_SPEC,
        "person_selection": {"rules": selection["rules"], "targets": selection["targets"],
                             "rule_counts": selection["rule_counts"],
                             "annotations_zip_sha256": selection["annotations_zip_sha256"]},
        "compass": ({"dir": compass_dir.name, "version_info": compass_version,
                     "duplicate_copies_dropped": rock_dups} if compass_dir else None),
        "drops": {k: dict(v) for k, v in drops.items()},
        "excluded_ids_file": args.exclude_ids,
        "stats": stats,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    (out_root / "README.md").write_text(build_readme(out_root.name, meta))

    print(f"package -> {out_root}")
    for name, value in stats.items():
        print(f"  {name}: {value}")
    print(f"  drops: {meta['drops']}  compass duplicate copies: {rock_dups}")
    print(f"next: python scripts/04_verify_package.py --package {out_root}")


def build_readme(pkg_name, meta):
    s = meta["stats"]
    rows = "\n".join(f"| `{name}.jsonl` | {v['samples']} | {v['pos']} | {v['neg']} | {v['boxes']} | {v['max_boxes_per_image']} |"
                     for name, v in s.items() if "pos" in v)
    has_rock = "rock" in meta["classes"]
    mix = f"person_train + rock_train × {meta['rock_repeat']}" if has_rock else "person_train"
    rock_source = ""
    if has_rock:
        compass_v = (meta["compass"]["version_info"] or {}).get("id", meta["compass"]["dir"])
        rock_source = (f"\n- **rock**: Compass Rock Detection (Roboflow `compass-dhncp/rock-detection-9kgmi`, "
                       f"{compass_v}); Roboflow train/valid/test kept as train/val/test; images without boxes are "
                       "negatives; augmented copies of one original never straddle splits.")
    return f"""# {pkg_name}: Qwen3-VL grounding SFT data package ({', '.join(meta['classes'])})

Data only. Training pipelines for ms-swift, TRL+PEFT, NVIDIA NeMo AutoModel, Axolotl and Unsloth (and the
shared evaluation) live in `vlm_prep/pipelines/` -- see `pipelines/README.md`. ms-swift consumes these jsonl
files directly (with `ROOT_IMAGE_DIR=<this folder>`); the other frameworks convert them with their
`pipelines/<fw>/prep_data.py`.

## Files

| file | samples | pos | neg | boxes | max boxes/img |
|---|---|---|---|---|---|
{rows}

- `train.jsonl` = {mix} (seeded shuffle): {s['train_mix']}
- `val.jsonl`: {s['val_mix']}
- `*_test.jsonl`: held-out test sets for final evaluation (never used in training)
- `records/<name>.flat.jsonl`: same row order as `<name>.jsonl`, with `id`, `width/height`,
  `boxes_pixel` (xyxy px), `boxes_1000`, source ids -- used by the evaluation
- `records/token_lengths.json` (written by 04): per-sample total / image token counts
- `meta.json`: seed, selection rules, quantization spec, drop counts, stats

## Row format (ms-swift `messages` + `images`; image paths relative to this folder)

```json
{{"messages": [
  {{"role": "user", "content": "<image>Locate every instance that belongs to the following categories: \\"person\\". Report bbox coordinates in JSON format."}},
  {{"role": "assistant", "content": "```json\\n[\\n\\t{{\\"bbox_2d\\": [x1, y1, x2, y2], \\"label\\": \\"person\\"}}\\n]\\n```"}}],
 "images": ["images/coco/train2017/000000xxxxxx.jpg"]}}
```

- Prompt and answer follow the official Qwen3-VL 2d_grounding cookbook. Empty answer (negative image):
  `` ```json\\n[]\\n``` ``.
- Coordinates are **relative ints in [0,1000]** (Qwen3-VL convention): `pixel_x = x/1000*width`,
  `pixel_y = y/1000*height`, order x1,y1,x2,y2, boxes sorted ascending by (x1,y1,x2,y2). Any resize the
  processor does leaves labels valid. Boxes are relative to the raw stored pixels (images carrying an EXIF
  rotation tag were dropped).

## Sources and rules

- **person**: COCO 2017 (`train` from train2017; `val`/`test` disjoint subsets of val2017). Positives have
  1–{meta['person_selection']['rules']['max_objects']} person boxes and no `iscrowd` person region (crowd
  regions contain un-boxed people). Negatives have no person annotation (COCO labels its 80 classes exhaustively).{rock_source}
- Dropped images: {meta['drops']}.

## Known limitations

- Prompts are single-class only; no image is labeled for more than one class, so the package cannot train or
  evaluate a joint multi-class prompt.

## Licenses

- COCO 2017 annotations: CC BY 4.0; images: Flickr terms of use (https://cocodataset.org/#termsofuse).
- Compass Rock Detection (if included): CC BY 4.0 (https://universe.roboflow.com/compass-dhncp/rock-detection-9kgmi).
- Qwen3-VL weights: Apache-2.0.

Built {meta['created_at']}, seed {meta['seed']}.
"""


if __name__ == "__main__":
    main()
