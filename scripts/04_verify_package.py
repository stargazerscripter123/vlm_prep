"""
Step 04: verify the ms-swift person/rock package end-to-end. Non-zero exit on any failure.

Checks:
  1. schema: each <name>.jsonl row == swift_row(flat record) (same order); one user + one
     assistant turn, exactly one <image> tag, exactly one image
  2. images: exist under the package root, PIL size == declared, no EXIF rotation, decodable
  3. answer round-trip: parse(fenced JSON) == boxes_1000 (order included), labels == prompt
     class, negatives are exactly [] ; prompt names the record's class
  4. quantization: re-quantized boxes_pixel == boxes_1000 (+-1), inverse error <= D/2000,
     values in [0,1000], x1<x2, y1<y2, sorted
  5. splits: train/val mixes trace back to per-source rows (rock x rock_repeat); test ids
     unused elsewhere; no image content (sha1) and no Compass original stem in two splits
  6. tokens: Qwen3-VL processor (tokenizer/processor files only, no weights) -- analytic
     length for every sample, cross-checked against the real processor on a sample;
     max must be < --max-length
  7. previews: all rock images + a person sample, boxes drawn from boxes_1000 via the inverse
     transform (exactly what the model learns) -> data/preview/<pkg>/index.html
     (reject checkboxes export rejected_ids.txt for 03 --exclude-ids)
  8. --tar: on success write data/package/<pkg>.tar

Usage:
    python scripts/04_verify_package.py --package data/package/vlm_sft_person_rock_YYMMDD --tar
"""

import argparse
import hashlib
import html
import json
import random
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.setup_env()  # HF cache inside the project, before transformers is imported

from PIL import Image, ImageDraw  # noqa: E402

SOURCES = ("person", "rock")
SPLITS = ("train", "val", "test")
COLORS = {"person": (255, 40, 40), "rock": (40, 200, 255)}

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        if len(failures) <= 50:
            print(f"FAIL: {message}")
    return condition


def parse_args():
    parser = argparse.ArgumentParser(description="Verify ms-swift person/rock SFT package")
    parser.add_argument("--package", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--min-pixels", type=int, default=65536,
                        help="must match pipelines/common/hparams.sh MIN_PIXELS")
    parser.add_argument("--max-pixels", type=int, default=1048576,
                        help="must match pipelines/common/hparams.sh MAX_PIXELS")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--processor-samples", type=int, default=60)
    parser.add_argument("--skip-tokens", action="store_true")
    parser.add_argument("--person-previews", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tar", action="store_true")
    return parser.parse_args()


def load_sets(pkg):
    """Per-source sets that exist in the package (rock is optional)."""
    flat, swift = {}, {}
    for source in SOURCES:
        if not (pkg / f"{source}_train.jsonl").exists():
            continue
        for split in SPLITS:
            name = f"{source}_{split}"
            flat[name] = C.load_jsonl(pkg / "records" / f"{name}.flat.jsonl")
            swift[name] = C.load_jsonl(pkg / f"{name}.jsonl")
    return flat, swift


def verify_rows(flat, swift):
    for name in flat:
        check(len(flat[name]) == len(swift[name]), f"{name}: {len(flat[name])} flat != {len(swift[name])} swift rows")
        for record, row in zip(flat[name], swift[name]):
            rid = record["id"]
            check(row == C.swift_row(record["label"], record["image"], record["boxes_1000"]),
                  f"{name}/{rid}: swift row != rendered flat record")
            messages = row["messages"]
            check([m["role"] for m in messages] == ["user", "assistant"], f"{rid}: roles {[m['role'] for m in messages]}")
            check(messages[0]["content"].count(C.IMAGE_TAG) == 1, f"{rid}: <image> tag count != 1")
            check(C.IMAGE_TAG not in messages[1]["content"], f"{rid}: <image> in answer")
            check(len(row["images"]) == 1, f"{rid}: {len(row['images'])} images")
            check(f'"{record["label"]}"' in messages[0]["content"], f"{rid}: prompt does not name {record['label']}")
            check(record["split"] == name.split("_")[1], f"{rid}: split field {record['split']} in {name}")
            verify_answer(record)


def verify_answer(record):
    rid = record["id"]
    try:
        parsed = C.parse_answer(record["target"])
    except Exception as exc:
        check(False, f"{rid}: answer does not parse ({exc})")
        return
    check([box for _, box in parsed] == record["boxes_1000"], f"{rid}: parsed boxes != boxes_1000")
    check(all(label == record["label"] for label, _ in parsed), f"{rid}: label mismatch in answer")
    if record["is_negative"]:
        check(record["target"] == "```json\n[]\n```", f"{rid}: negative answer not exact")
        check(record["boxes_pixel"] == [] and record["boxes_1000"] == [], f"{rid}: negative with boxes")
        return
    check(len(record["boxes_pixel"]) == len(record["boxes_1000"]) > 0, f"{rid}: box arrays inconsistent/empty")
    check([tuple(b) for b in record["boxes_1000"]] == sorted(tuple(b) for b in record["boxes_1000"]),
          f"{rid}: boxes not sorted")
    width, height = record["width"], record["height"]
    for pixel, quantized in zip(record["boxes_pixel"], record["boxes_1000"]):
        check(all(0 <= v <= 1000 for v in quantized), f"{rid}: coord out of range {quantized}")
        check(quantized[0] < quantized[2] and quantized[1] < quantized[3], f"{rid}: degenerate {quantized}")
        requantized = [C.quantize(pixel[0], width), C.quantize(pixel[1], height),
                       C.quantize(pixel[2], width), C.quantize(pixel[3], height)]
        check(all(abs(a - b) <= 1 for a, b in zip(requantized, quantized)),
              f"{rid}: re-quantization {requantized} != {quantized}")
        for axis, (q, denom) in enumerate(zip(quantized, [width, height] * 2)):
            error = abs(q / 1000 * denom - pixel[axis])
            check(error <= denom / 2000 + 0.02, f"{rid}: inverse error {error:.3f}px axis {axis}")


def verify_images(pkg, flat):
    """Returns {image_rel: sha1}."""
    digests = {}
    for rows in flat.values():
        for record in rows:
            rel = record["image"]
            if rel in digests:
                continue
            path = pkg / rel
            if not check(path.is_file() and not rel.startswith("/"), f"{record['id']}: image missing {rel}"):
                continue
            digests[rel] = hashlib.sha1(path.read_bytes()).hexdigest()
            try:
                with Image.open(path) as image:
                    check(image.size == (record["width"], record["height"]),
                          f"{record['id']}: size {image.size} != {(record['width'], record['height'])}")
                    check(C.exif_orientation(image) in (None, 1), f"{record['id']}: EXIF rotation tag")
                    image.load()
            except Exception as exc:
                check(False, f"{record['id']}: decode error {exc}")
    return digests


def verify_splits(pkg, flat, swift, digests, rock_repeat):
    by_id = {r["id"]: r for rows in flat.values() for r in rows}
    check(len(by_id) == sum(len(rows) for rows in flat.values()), "duplicate sample ids across sets")

    has_rock = "rock_train" in flat
    mixes = (("train", {"person_train": 1, **({"rock_train": rock_repeat} if has_rock else {})}),
             ("val", {"person_val": 1, **({"rock_val": 1} if has_rock else {})}))
    for mix, parts in mixes:
        rows = C.load_jsonl(pkg / f"{mix}.jsonl")
        ids = (pkg / "records" / f"{mix}.ids.txt").read_text().split()
        check(len(rows) == len(ids), f"{mix}.jsonl: {len(rows)} rows != {len(ids)} ids")
        for row, rid in zip(rows, ids):
            record = by_id.get(rid)
            if check(record is not None, f"{mix}: unknown id {rid}"):
                check(row == C.swift_row(record["label"], record["image"], record["boxes_1000"]),
                      f"{mix}: row for {rid} != its source row")
        expected = Counter()
        for name, times in parts.items():
            for record in flat[name]:
                expected[record["id"]] += times
        check(Counter(ids) == expected, f"{mix}.jsonl id multiset != sources {parts}")

    split_of_digest, split_of_stem = defaultdict(set), defaultdict(set)
    for name, rows in flat.items():
        split = name.split("_")[1]
        for record in rows:
            if record["image"] in digests:
                split_of_digest[digests[record["image"]]].add(split)
            if record["label"] == "rock":
                split_of_stem[record["source"]["original_stem"]].add(split)
    leaks = [s for s in split_of_digest.values() if len(s) > 1]
    stem_leaks = [k for k, s in split_of_stem.items() if len(s) > 1]
    coco_ids = {split: {r["source"]["image_id"] for r in flat[f"person_{split}"]} for split in SPLITS}
    ok = check(not leaks, f"{len(leaks)} identical images appear in more than one split")
    ok &= check(not stem_leaks, f"compass original stems in more than one split: {stem_leaks[:5]}")
    ok &= check(not coco_ids["val"] & coco_ids["test"], "COCO val/test image ids overlap")
    if ok:
        print(f"splits ok: {len(digests)} unique images, no cross-split duplicates")


def verify_tokens(pkg, flat, args):
    from qwen_vl_utils import smart_resize
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    image_processor = processor.image_processor
    min_pixels, max_pixels = args.min_pixels, args.max_pixels
    image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    for attr, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        try:  # read-only property (derived from size) in some transformers versions
            setattr(image_processor, attr, value)
        except AttributeError:
            pass
    merge = image_processor.merge_size
    factor = image_processor.patch_size * merge
    tokenizer = processor.tokenizer
    image_pad = processor.image_token

    def hf_messages(record):
        prompt = C.build_prompt(record["label"]).replace(C.IMAGE_TAG, "", 1)
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": record["target"]}]}]

    records = [r for rows in flat.values() for r in rows]
    lengths, analytic, per_sample = [], {}, {}
    for record in records:
        text = processor.apply_chat_template(hf_messages(record), tokenize=False)
        h, w = smart_resize(record["height"], record["width"], factor=factor,
                            min_pixels=min_pixels, max_pixels=max_pixels)
        n_image = (h // factor) * (w // factor)
        n_text = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        analytic[record["id"]] = n_text - 1 + n_image  # one <|image_pad|> expands to n_image tokens
        per_sample[record["id"]] = {"total": analytic[record["id"]], "image": n_image,
                                    "grid_hw": [h // 16, w // 16]}
        lengths.append(analytic[record["id"]])

    rng = random.Random(args.seed)
    for record in rng.sample(records, min(args.processor_samples, len(records))):
        text = processor.apply_chat_template(hf_messages(record), tokenize=False)
        with Image.open(pkg / record["image"]) as image:
            inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        n_real = inputs["input_ids"].shape[1]
        check(n_real == analytic[record["id"]],
              f"{record['id']}: processor length {n_real} != analytic {analytic[record['id']]}")
        check(text.count(image_pad) == 1, f"{record['id']}: chat template image pad count != 1")

    lengths.sort()
    p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
    answer_tokens = [len(tokenizer(r["target"], add_special_tokens=False)["input_ids"]) for r in records]
    print(f"tokens (pixels {min_pixels}..{max_pixels}): p50 {statistics.median(lengths):.0f} "
          f"p99 {p99} max {lengths[-1]}; answer tokens max {max(answer_tokens)}")
    check(lengths[-1] < args.max_length, f"max sample length {lengths[-1]} >= --max-length {args.max_length}")
    stats = {"p50": statistics.median(lengths), "p99": p99, "max": lengths[-1],
             "answer_max": max(answer_tokens), "min_pixels": min_pixels, "max_pixels": max_pixels}
    return stats, per_sample


def draw_previews(pkg, flat, n_person, seed):
    out = C.PREVIEW_DIR / pkg.name
    C.assert_not_dropbox(out)
    rng = random.Random(seed)
    person_pos = [r for r in flat["person_train"] if not r["is_negative"]]
    person_neg = [r for r in flat["person_train"] if r["is_negative"]]
    chosen = ([r for s in SPLITS for r in flat.get(f"rock_{s}", [])]
              + rng.sample(person_pos, min(n_person, len(person_pos)))
              + rng.sample(person_neg, min(5, len(person_neg))))
    cards = []
    for record in chosen:
        dest = out / record["label"] / f"{record['id']}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(pkg / record["image"]) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            line = max(2, round(max(image.size) / 300))
            for x1, y1, x2, y2 in record["boxes_1000"]:
                draw.rectangle([x1 / 1000 * record["width"], y1 / 1000 * record["height"],
                                x2 / 1000 * record["width"], y2 / 1000 * record["height"]],
                               outline=COLORS[record["label"]], width=line)
            image.thumbnail((1024, 1024))
            image.save(dest, "JPEG", quality=85)
        cards.append((record, dest.relative_to(out).as_posix()))
    (out / "index.html").write_text(gallery_html(pkg.name, cards))
    print(f"{len(cards)} previews -> {out / 'index.html'}")
    return out


def gallery_html(pkg_name, cards):
    items = "\n".join(
        f'<figure data-id="{html.escape(r["id"])}"><img loading="lazy" src="{html.escape(src)}">'
        f'<figcaption><label><input type="checkbox"> reject</label> {html.escape(r["label"])} · {r["split"]} · '
        f'{len(r["boxes_1000"])} box · <code>{html.escape(r["id"])}</code></figcaption></figure>'
        for r, src in cards)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(pkg_name)} preview</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:13px system-ui;margin:16px;background:#111;color:#ddd}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;padding:6px;border-radius:6px}}figure.rej{{outline:3px solid #e33}}
img{{width:100%;display:block}}code{{font-size:11px;word-break:break-all}}
header{{position:sticky;top:0;background:#111;padding:8px 0;z-index:1}}</style></head><body>
<header><b>{html.escape(pkg_name)}</b> — boxes drawn from boxes_1000 (what the model learns).
Red = person, blue = rock. <button id="export">Export rejected ids</button> <span id="count"></span></header>
<main>{items}</main>
<script>
const KEY = "reject:{html.escape(pkg_name)}";
let rejected = new Set();
try {{ rejected = new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); }} catch (e) {{}}
const count = document.getElementById("count");
function sync() {{ count.textContent = rejected.size + " rejected";
  try {{ localStorage.setItem(KEY, JSON.stringify([...rejected])); }} catch (e) {{}} }}
document.querySelectorAll("figure").forEach(fig => {{
  const box = fig.querySelector("input"), id = fig.dataset.id;
  box.checked = rejected.has(id); fig.classList.toggle("rej", box.checked);
  box.addEventListener("change", () => {{ box.checked ? rejected.add(id) : rejected.delete(id);
    fig.classList.toggle("rej", box.checked); sync(); }});
}});
document.getElementById("export").onclick = () => {{
  const blob = new Blob([[...rejected].join("\\n") + "\\n"], {{type: "text/plain"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "rejected_ids.txt"; a.click(); }};
sync();
</script></body></html>
"""


def write_tar(pkg):
    tar_path = pkg.parent / f"{pkg.name}.tar"
    tmp = tar_path.with_suffix(".tar.part")
    with tarfile.open(tmp, "w") as tar:
        tar.add(pkg, arcname=pkg.name)
    tmp.replace(tar_path)
    print(f"tar -> {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")


def main():
    args = parse_args()
    pkg = Path(args.package).resolve()
    C.assert_not_dropbox(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    flat, swift = load_sets(pkg)
    print({name: len(rows) for name, rows in flat.items()})

    verify_rows(flat, swift)
    digests = verify_images(pkg, flat)
    verify_splits(pkg, flat, swift, digests, meta["rock_repeat"])
    token_stats, per_sample = (None, None) if args.skip_tokens else verify_tokens(pkg, flat, args)
    draw_previews(pkg, flat, args.person_previews, args.seed)

    if failures:
        print(f"\nVERIFY FAILED: {len(failures)} problem(s)")
        sys.exit(1)
    if token_stats:
        (pkg / "records" / "token_stats.json").write_text(json.dumps(token_stats, indent=2))
        (pkg / "records" / "token_lengths.json").write_text(json.dumps(per_sample))
    print("\nVERIFY OK: all checks passed")
    if args.tar:
        write_tar(pkg)


if __name__ == "__main__":
    main()
