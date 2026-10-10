"""
Framework-neutral view of the data package, shared by every pipelines/<fw>/prep_data.py and the eval.

A package row (ms-swift format) is {"messages":[user "<image>PROMPT", assistant ANSWER], "images":[rel]}.
Here every row becomes a Sample(id, image=ABSOLUTE path, prompt=PROMPT without <image>, answer, ...),
cross-checked against records/*.flat.jsonl so ids, images and answers can never drift apart.

Splits:
  train / val      -> package mixes (train.jsonl/val.jsonl, ids in records/<split>.ids.txt)
  person_test, ... -> per-source sets (<name>.jsonl, same order as records/<name>.flat.jsonl)
  audit32          -> fixed, seeded 32-sample subset of train (28 positives incl. the most crowded
                      image + 4 negatives) used by every framework's label audit and step-1 loss check
"""

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

IMAGE_TAG = "<image>"
AUDIT_SEED = 20261009
AUDIT_POS, AUDIT_NEG = 28, 4


@dataclass(frozen=True)
class Sample:
    id: str
    image: str          # absolute path
    prompt: str         # user text, no <image> tag
    answer: str         # assistant text (fenced JSON)
    label: str
    width: int
    height: int
    is_negative: bool
    n_boxes: int


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _flat_index(pkg):
    by_id = {}
    for path in sorted((pkg / "records").glob("*.flat.jsonl")):
        for record in _load_jsonl(path):
            by_id[record["id"]] = record
    return by_id


def _to_sample(pkg, row, record):
    user, assistant = row["messages"]
    if not user["content"].startswith(IMAGE_TAG) or row["images"] != [record["image"]]:
        raise ValueError(f"{record['id']}: package row does not match its flat record")
    if assistant["content"] != record["target"]:
        raise ValueError(f"{record['id']}: answer differs from flat record target")
    return Sample(
        id=record["id"],
        image=str((pkg / record["image"]).resolve()),
        prompt=user["content"][len(IMAGE_TAG):],
        answer=assistant["content"],
        label=record["label"],
        width=record["width"],
        height=record["height"],
        is_negative=record["is_negative"],
        n_boxes=len(record["boxes_1000"]),
    )


def load_split(pkg, split):
    pkg = Path(pkg).resolve()
    if split == "audit32":
        return audit_subset(load_split(pkg, "train"))
    rows = _load_jsonl(pkg / f"{split}.jsonl")
    ids_file = pkg / "records" / f"{split}.ids.txt"
    if ids_file.exists():                       # mixed train/val
        ids = ids_file.read_text().split()
        flat = _flat_index(pkg)
        records = [flat[i] for i in ids]
    else:                                       # per-source set, same row order as its flat file
        records = _load_jsonl(pkg / "records" / f"{split}.flat.jsonl")
    if len(rows) != len(records):
        raise ValueError(f"{split}: {len(rows)} rows vs {len(records)} records")
    return [_to_sample(pkg, row, record) for row, record in zip(rows, records)]


def audit_subset(train):
    """Deterministic 32-sample subset: the most crowded positive + 27 seeded positives + 4 negatives."""
    positives = sorted((s for s in train if not s.is_negative), key=lambda s: s.id)
    negatives = sorted((s for s in train if s.is_negative), key=lambda s: s.id)
    crowded = max(positives, key=lambda s: (s.n_boxes, s.id))
    rng = random.Random(AUDIT_SEED)
    rest = [s for s in positives if s.id != crowded.id]
    chosen = [crowded] + rng.sample(rest, AUDIT_POS - 1) + rng.sample(negatives, AUDIT_NEG)
    return sorted(chosen, key=lambda s: s.id)


EVAL_SEED = 20261010
EVAL_SUBSETS = {"n100": (90, 10), "n50": (45, 5)}   # (positives, negatives); "full" = everything


def eval_subset(samples, name):
    """Fixed, seeded, stratified evaluation subset shared by every framework (and the baselines)."""
    if name == "full":
        return list(samples)
    n_pos, n_neg = EVAL_SUBSETS[name]
    rng = random.Random(EVAL_SEED)
    positives = sorted((s for s in samples if not s.is_negative), key=lambda s: s.id)
    negatives = sorted((s for s in samples if s.is_negative), key=lambda s: s.id)
    chosen = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    return sorted(chosen, key=lambda s: s.id)


def fingerprint(pkg, splits, code=()):
    """Changes whenever the package location, any source file of the given splits, or the converter
    code (the calling prep_data.py and this module) changes."""
    pkg = Path(pkg).resolve()
    digest = hashlib.sha256(str(pkg).encode())
    for split in splits:
        name = "train" if split == "audit32" else split
        for path in (pkg / f"{name}.jsonl", pkg / "records" / f"{name}.ids.txt",
                     pkg / "records" / f"{name}.flat.jsonl"):
            if path.exists():
                digest.update(path.read_bytes())
    for path in (Path(__file__), *code):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def assert_not_dropbox(path):
    resolved = str(Path(path).resolve())
    if "Dropbox" in resolved:
        sys.exit(f"refusing to write inside Dropbox: {resolved}")


DEFAULT_SPLITS = ("train", "val", "person_test", "audit32")


def prep_main(framework, convert, argv=None):
    """Common CLI for pipelines/<fw>/prep_data.py: convert each split with `convert(sample) -> dict`.

    Writes <out>/<split>.jsonl + <split>.ids.txt + prep_meta.json; skips when the fingerprint is
    unchanged (use --force to rebuild).
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"Convert the data package for {framework}")
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    pkg, out = Path(args.pkg).resolve(), Path(args.out).resolve()
    assert_not_dropbox(out)
    out.mkdir(parents=True, exist_ok=True)
    splits = [s for s in args.splits if s == "audit32" or (pkg / f"{s}.jsonl").exists()]
    meta_path = out / "prep_meta.json"
    fp = fingerprint(pkg, splits, code=[sys.argv[0]])
    if not args.force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp and all((out / f"{s}.jsonl").exists() for s in splits):
            print(f"[prep {framework}] up to date: {out}")
            return

    counts = {}
    for split in splits:
        samples = load_split(pkg, split)
        write_jsonl(out / f"{split}.jsonl", [convert(s) for s in samples])
        (out / f"{split}.ids.txt").write_text("\n".join(s.id for s in samples) + "\n")
        counts[split] = len(samples)
    meta_path.write_text(json.dumps({"framework": framework, "package": str(pkg), "fingerprint": fp,
                                     "counts": counts}, indent=2))
    print(f"[prep {framework}] {counts} -> {out}")
