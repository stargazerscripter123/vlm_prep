"""
Step 02: Compass rock detection (Roboflow Universe compass-dhncp/rock-detection-9kgmi, CC BY 4.0).

API key: $ROBOFLOW_API_KEY, else ROBOFLOW_API_KEY=... in <project>/.env (chmod 600).
The key is never printed or written anywhere by this script.

  --list-versions   print per-version image count, splits, preprocessing, augmentation
  --version N       download version N as COCO JSON -> data/raw/compass/v<N>/{train,valid,test}/

Pick a version without augmentation and without "stretch" resize when one exists
(augmented copies of one original must never straddle splits; step 03 checks this).

Usage:
    python scripts/02_download_compass.py --list-versions
    python scripts/02_download_compass.py --version 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WORKSPACE = "compass-dhncp"
PROJECT = "rock-detection-9kgmi"
INFO_KEYS = ("id", "name", "created", "images", "splits", "preprocessing", "augmentation", "exports")


def parse_args():
    parser = argparse.ArgumentParser(description="Download Compass rock detection from Roboflow")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-versions", action="store_true")
    group.add_argument("--version", type=int)
    parser.add_argument("--out", type=str, default=str(C.RAW_DIR / "compass"))
    return parser.parse_args()


def load_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    env_file = C.PROJECT_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.strip().partition("=")
            if name == "ROBOFLOW_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        sys.exit(f"ROBOFLOW_API_KEY not set (export it or put ROBOFLOW_API_KEY=... in {env_file})")
    return key


def main():
    args = parse_args()
    from roboflow import Roboflow

    project = Roboflow(api_key=load_api_key()).workspace(WORKSPACE).project(PROJECT)
    versions = project.get_version_information()
    summaries = [{k: v.get(k) for k in INFO_KEYS} for v in versions]

    if args.list_versions:
        print(json.dumps(summaries, indent=2, default=str))
        return

    matches = [s for s in summaries if str(s["id"]).rstrip("/").split("/")[-1] == str(args.version)]
    if not matches:
        sys.exit(f"version {args.version} not found; available: {[s['id'] for s in summaries]}")
    out_dir = Path(args.out).resolve() / f"v{args.version}"
    C.assert_not_dropbox(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    project.version(args.version).download("coco", location=str(out_dir), overwrite=False)
    (out_dir / "version_info.json").write_text(json.dumps(matches[0], indent=2, default=str))

    for split in ("train", "valid", "test"):
        ann = out_dir / split / "_annotations.coco.json"
        if ann.exists():
            payload = json.loads(ann.read_text())
            print(f"{split}: {len(payload['images'])} images, {len(payload['annotations'])} boxes, "
                  f"categories={[c['name'] for c in payload['categories']]}")
        else:
            print(f"{split}: (missing)")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
