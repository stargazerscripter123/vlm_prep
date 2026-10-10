"""
Step 01: COCO 2017 person subset -- download annotations, select images, download only those.

Selection (category_id 1 = person):
  pos: 1..--max-objects person boxes and NO person annotation with iscrowd=1
       (a crowd region means un-boxed people -> would teach "absent")
  neg: zero person annotations (COCO is exhaustively labeled for its 80 classes)
Candidates are a seeded shuffle of sorted ids; ~--extra-frac extra are downloaded so
step 03 can take the first N that pass its image checks.
  train <- train2017 : --person-train pos + --neg-frac negatives
  val   <- val2017   : --person-val pos (+neg), test <- val2017: --person-test pos (+neg), disjoint

Outputs (data/raw/coco/):
  annotations_trainval2017.zip, annotations/instances_{train,val}2017.json
  {train2017,val2017}/<file_name>       downloaded images
  selection.json                        ordered candidates with person boxes + rule stats
  download_report.jsonl

Usage:
    python scripts/01_download_coco.py [--person-train 5000] [--workers 16]
"""

import argparse
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"
ANN_ZIP_URL = f"{BASE_URL}/annotations/annotations_trainval2017.zip"
PERSON_CATEGORY_ID = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Download COCO 2017 person subset")
    parser.add_argument("--out", type=str, default=str(C.RAW_DIR / "coco"))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--person-train", type=int, default=5000)
    parser.add_argument("--person-val", type=int, default=300)
    parser.add_argument("--person-test", type=int, default=1000)
    parser.add_argument("--neg-frac", type=float, default=0.1)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--extra-frac", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--select-only", action="store_true", help="write selection.json, skip image download")
    return parser.parse_args()


def fetch_annotations(out_root):
    zip_path = out_root / "annotations_trainval2017.zip"
    with requests.Session() as session:
        result = C.download_file(session, ANN_ZIP_URL, zip_path, expect_image=False, timeout=300)
    if result["status"] == "failed":
        sys.exit(f"annotation download failed: {result['error']}")
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(exist_ok=True)
    paths = {}
    with zipfile.ZipFile(zip_path) as archive:
        for split in ("train2017", "val2017"):
            member = f"annotations/instances_{split}.json"
            target = ann_dir / f"instances_{split}.json"
            if not target.exists():
                archive.extract(member, out_root)
            paths[split] = target
    return zip_path, paths


def classify(instances_path, max_objects):
    """Return {image_id: entry} for pos/neg candidates plus rule counts."""
    payload = json.loads(instances_path.read_text())
    images = {image["id"]: image for image in payload["images"]}
    person_anns = defaultdict(list)
    for ann in payload["annotations"]:
        if ann["category_id"] == PERSON_CATEGORY_ID:
            person_anns[ann["image_id"]].append(ann)

    pos, neg = {}, {}
    counts = {"images": len(images), "pos": 0, "neg": 0, "excluded_crowd": 0, "excluded_too_many": 0}
    for image_id, image in images.items():
        anns = person_anns.get(image_id, [])
        entry = {"id": image_id, "file_name": image["file_name"],
                 "width": image["width"], "height": image["height"]}
        if not anns:
            neg[image_id] = {**entry, "boxes_xywh": [], "ann_ids": []}
            counts["neg"] += 1
        elif any(a["iscrowd"] for a in anns):
            counts["excluded_crowd"] += 1
        elif len(anns) > max_objects:
            counts["excluded_too_many"] += 1
        else:
            anns = sorted(anns, key=lambda a: a["id"])
            pos[image_id] = {**entry, "boxes_xywh": [a["bbox"] for a in anns],
                             "ann_ids": [a["id"] for a in anns]}
            counts["pos"] += 1
    return pos, neg, counts


def shuffled(ids, seed):
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def with_extra(n, extra_frac):
    return math.ceil(n * (1 + extra_frac))


def main():
    args = parse_args()
    out_root = Path(args.out).resolve()
    C.assert_not_dropbox(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    zip_path, ann_paths = fetch_annotations(out_root)
    print(f"annotations ready: {zip_path.name}")

    train_pos, train_neg, train_counts = classify(ann_paths["train2017"], args.max_objects)
    val_pos, val_neg, val_counts = classify(ann_paths["val2017"], args.max_objects)
    print(f"train2017 rules: {train_counts}")
    print(f"val2017   rules: {val_counts}")

    def n_neg(n_pos):
        return round(n_pos * args.neg_frac)

    # Candidate lists hold ~extra_frac more than needed; step 03 takes the first N passing checks.
    train_pos_ids = shuffled(train_pos, args.seed)[:with_extra(args.person_train, args.extra_frac)]
    train_neg_ids = shuffled(train_neg, args.seed)[:with_extra(n_neg(args.person_train), args.extra_frac)]
    val_pos_order = shuffled(val_pos, args.seed)
    val_neg_order = shuffled(val_neg, args.seed)
    n_vp = with_extra(args.person_val, args.extra_frac)
    n_tp = with_extra(args.person_test, args.extra_frac)
    n_vn = with_extra(n_neg(args.person_val), args.extra_frac)
    n_tn = with_extra(n_neg(args.person_test), args.extra_frac)
    if n_vp + n_tp > len(val_pos_order) or n_vn + n_tn > len(val_neg_order):
        sys.exit("val2017 too small for requested val+test sizes")

    candidates = {
        "train": {"coco_split": "train2017",
                  "pos": [train_pos[i] for i in train_pos_ids],
                  "neg": [train_neg[i] for i in train_neg_ids]},
        "val": {"coco_split": "val2017",
                "pos": [val_pos[i] for i in val_pos_order[:n_vp]],
                "neg": [val_neg[i] for i in val_neg_order[:n_vn]]},
        "test": {"coco_split": "val2017",
                 "pos": [val_pos[i] for i in val_pos_order[n_vp:n_vp + n_tp]],
                 "neg": [val_neg[i] for i in val_neg_order[n_vn:n_vn + n_tn]]},
    }
    val_ids = {e["id"] for k in ("pos", "neg") for e in candidates["val"][k]}
    test_ids = {e["id"] for k in ("pos", "neg") for e in candidates["test"][k]}
    assert not val_ids & test_ids, "val/test overlap"

    selection = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "rules": {"category_id": PERSON_CATEGORY_ID, "max_objects": args.max_objects,
                  "exclude_iscrowd": True, "neg_frac": args.neg_frac, "extra_frac": args.extra_frac},
        "targets": {"train": {"pos": args.person_train, "neg": n_neg(args.person_train)},
                    "val": {"pos": args.person_val, "neg": n_neg(args.person_val)},
                    "test": {"pos": args.person_test, "neg": n_neg(args.person_test)}},
        "rule_counts": {"train2017": train_counts, "val2017": val_counts},
        "annotations_zip_sha256": C.sha256_file(zip_path),
        "candidates": candidates,
    }
    (out_root / "selection.json").write_text(json.dumps(selection, indent=1))
    for split, block in candidates.items():
        print(f"{split}: {len(block['pos'])} pos + {len(block['neg'])} neg candidates")
    if args.select_only:
        return

    tasks = {}
    for block in candidates.values():
        for kind in ("pos", "neg"):
            for entry in block[kind]:
                rel = f"{block['coco_split']}/{entry['file_name']}"
                tasks[rel] = (f"{BASE_URL}/{rel}", out_root / rel)
    print(f"downloading {len(tasks)} images with {args.workers} workers ...")

    results = []
    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
        session.mount("https://", adapter)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(C.download_file, session, url, dest) for url, dest in tasks.values()]
            for future in tqdm(as_completed(futures), total=len(futures), unit="img"):
                results.append(future.result())
    C.write_jsonl(out_root / "download_report.jsonl", results)

    by_status = defaultdict(int)
    for result in results:
        by_status[result["status"]] += 1
    total_bytes = sum(r.get("bytes", 0) for r in results)
    print(f"download status: {dict(by_status)}  ({total_bytes / 1e9:.2f} GB on disk)")
    if by_status["failed"]:
        sys.exit(f"{by_status['failed']} downloads failed -- rerun to resume (see download_report.jsonl)")


if __name__ == "__main__":
    main()
