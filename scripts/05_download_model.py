"""
Step 05: download a base model (default Qwen3-VL-32B-Instruct, BF16, ~67 GB) to models/<name>/.

Pinned to --revision; after download every file is checked against the Hub metadata
(size, and sha256 for LFS files) and a `sha256sum -c`-compatible SHA256SUMS is written
into the model dir so the copy can be re-verified after rsync to the H200.

The pipelines find it as models/<name>/ (MODEL=4b|32b); `bash pipelines/run.sh setup` calls this script.

Usage:
    python scripts/05_download_model.py [--repo Qwen/Qwen3-VL-4B-Instruct] [--workers 8]
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.setup_env()
_user_token = Path.home() / ".cache" / "huggingface" / "token"
if _user_token.exists():
    os.environ.setdefault("HF_TOKEN_PATH", str(_user_token))  # reuse the existing HF login (not printed)

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen3-VL-32B-Instruct"
PINNED_REVISIONS = {       # the revisions every result of this project was produced with
    "Qwen/Qwen3-VL-32B-Instruct": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and verify the base VLM")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--revision", type=str, default=None,
                        help="default: the pinned revision of --repo (see PINNED_REVISIONS), else 'main'")
    parser.add_argument("--out", type=str, default=None, help="default: models/<repo name>")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.revision is None:
        args.revision = PINNED_REVISIONS.get(args.repo, "main")
        if args.repo not in PINNED_REVISIONS:
            print(f"warning: no pinned revision for {args.repo}; using 'main'")
    out = Path(args.out or C.PROJECT_ROOT / "models" / args.repo.split("/")[-1]).resolve()
    C.assert_not_dropbox(out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        print(f"downloading {args.repo}@{args.revision[:10]} -> {out}")
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(out),
                          max_workers=args.workers)

    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    problems, sums, total = [], [], 0
    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        path = out / sibling.rfilename
        if not path.is_file():
            problems.append(f"missing {sibling.rfilename}")
            continue
        size = path.stat().st_size
        total += size
        if sibling.size is not None and size != sibling.size:
            problems.append(f"{sibling.rfilename}: size {size} != hub {sibling.size}")
            continue
        digest = sha256_of(path)
        if sibling.lfs is not None and digest != sibling.lfs.sha256:
            problems.append(f"{sibling.rfilename}: sha256 mismatch")
        sums.append(f"{digest}  {sibling.rfilename}")
        print(f"  ok {size / 1e9:7.2f} GB  {sibling.rfilename}")

    if problems:
        print("\nMODEL VERIFY FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "download_info.json").write_text(json.dumps({
        "repo": args.repo, "revision": args.revision, "files": len(sums),
        "total_bytes": total, "verified_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"\nMODEL OK: {len(sums)} files, {total / 1e9:.2f} GB, sha256 verified -> {out}")
    print(f"on the H200: cd {out.name} && sha256sum -c SHA256SUMS")


if __name__ == "__main__":
    main()
