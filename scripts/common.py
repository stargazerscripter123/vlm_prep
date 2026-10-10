"""
Shared helpers for the Qwen3-VL person/rock grounding SFT package.

quantize()/quantize_box() are vendored from yoke/tools/vlm_sft/sft_common.py
(half-up, clamp to [0,1000], reject degenerate). The answer wire format follows
the official Qwen3-VL 2d_grounding cookbook:

  prompt : Locate every instance that belongs to the following categories: "person". Report bbox coordinates in JSON format.
  answer : ```json\n[\n\t{"bbox_2d": [x1, y1, x2, y2], "label": "person"},\n\t...\n]\n```
  empty  : ```json\n[]\n```
  coords : ints in [0,1000], relative per axis (pixel_x = x/1000*width, pixel_y = y/1000*height)
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PACKAGE_DIR = PROJECT_ROOT / "data" / "package"
PREVIEW_DIR = PROJECT_ROOT / "data" / "preview"
CACHE_DIR = PROJECT_ROOT / ".cache"

SEED = 20260930
IMAGE_TAG = "<image>"
PROMPT_TEMPLATE = ('Locate every instance that belongs to the following categories: "{label}". '
                   'Report bbox coordinates in JSON format.')

QUANT_SPEC = {
    "formula": "Q(v, D) = min(1000, max(0, floor(v / D * 1000 + 0.5)))  # half-up",
    "x_denominator": "image width (PIL-measured raw file, no EXIF transpose)",
    "y_denominator": "image height",
    "inverse": "pixel = q / 1000 * D",
    "box_order": "x1, y1, x2, y2",
    "multi_box_sort": "ascending by quantized (x1, y1, x2, y2) tuple",
    "max_roundtrip_error_px": "D / 2000",
}

_FENCE_RE = re.compile(r"^```json\n(.*)\n```$", re.S)


def setup_env():
    """Keep every cache inside the project (never Dropbox). Call before importing transformers/torch."""
    for key, sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("TMPDIR", "tmp")):
        path = CACHE_DIR / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def assert_not_dropbox(path):
    resolved = str(Path(path).resolve())
    assert "Dropbox" not in resolved, f"Refusing to write inside Dropbox: {resolved}"


def quantize(value, denominator):
    """Half-up quantization of one coordinate to the [0, 1000] range."""
    return min(1000, max(0, math.floor(value / denominator * 1000 + 0.5)))


def quantize_box(bbox_xyxy, width, height):
    """Pixel xyxy -> quantized [0,1000] ints. Normalizes order, clamps, rejects degenerate."""
    x1, y1, x2, y2 = bbox_xyxy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    quantized = [quantize(x1, width), quantize(y1, height),
                 quantize(x2, width), quantize(y2, height)]
    if quantized[0] == quantized[2] or quantized[1] == quantized[3]:
        raise ValueError(f"degenerate box after quantization: {bbox_xyxy} -> {quantized}")
    return [x1, y1, x2, y2], quantized


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def build_prompt(label):
    return IMAGE_TAG + PROMPT_TEMPLATE.format(label=label)


def render_answer(label, boxes_1000):
    """Cookbook-style fenced JSON list; boxes must already be sorted."""
    if not boxes_1000:
        return "```json\n[]\n```"
    lines = [f'\t{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}], "label": "{label}"}}'
             for x1, y1, x2, y2 in boxes_1000]
    return "```json\n[\n" + ",\n".join(lines) + "\n]\n```"


def parse_answer(text):
    """Inverse of render_answer: list of (label, [x1, y1, x2, y2]). Raises on malformed text."""
    match = _FENCE_RE.match(text)
    if match is None:
        raise ValueError("answer is not a ```json fenced block")
    items = json.loads(match.group(1))
    return [(item["label"], [int(v) for v in item["bbox_2d"]]) for item in items]


def swift_row(label, image_rel, boxes_1000):
    """One ms-swift training row: messages + images (image path relative to package root)."""
    return {
        "messages": [
            {"role": "user", "content": build_prompt(label)},
            {"role": "assistant", "content": render_answer(label, boxes_1000)},
        ],
        "images": [image_rel],
    }


def exif_orientation(image):
    """EXIF orientation tag (1 = normal) or None when absent."""
    return image.getexif().get(0x0112)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MAGIC = ((b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"))


def download_file(session, url, dest, retries=4, expect_image=True, timeout=60):
    """Atomic resumable download: skip non-empty dest, write .part, verify magic, os.replace."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "dest": str(dest), "status": "skipped", "bytes": dest.stat().st_size}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    error = None
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(1 << 20):
                        f.write(chunk)
            if expect_image:
                with open(tmp, "rb") as f:
                    head = f.read(12)
                if not any(head.startswith(magic) for magic, _ in _MAGIC):
                    raise ValueError(f"not an image (magic {head[:4]!r})")
            os.replace(tmp, dest)
            return {"url": url, "dest": str(dest), "status": "ok", "bytes": dest.stat().st_size}
        except Exception as exc:  # retried below
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    if tmp.exists():
        tmp.unlink()
    return {"url": url, "dest": str(dest), "status": "failed", "error": error}
