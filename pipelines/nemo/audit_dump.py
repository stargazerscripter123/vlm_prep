"""
NeMo AutoModel audit dump (NeMo does not use transformers.Trainer, so the shared hook cannot see its
batches): collate every audit32 row with the SAME collate function and processor settings as the
recipe and write <run>/audit_dump.json in the schema compare_audit.py expects. NeMo's collate
returns shifted tensors (input_ids[:, :-1], labels[:, 1:]); compare_audit accepts the one-token-shorter
input ids, and the trained ids (= label values) are directly comparable.
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoProcessor
from nemo_automodel.components.datasets.vlm.collate_fns import qwen2_5_collate_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_rock_data import make_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-pixels", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, size={"shortest_edge": args.min_pixels,
                                                                "longest_edge": args.max_pixels})
    tok = processor.tokenizer
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    samples = []
    for row in make_dataset(args.data):
        batch = qwen2_5_collate_fn([row], processor)
        ids, labels = batch["input_ids"][0], batch["labels"][0]
        trained = (labels != -100).nonzero().flatten()
        pixel = batch["pixel_values"].float()
        samples.append({
            "input_ids": ids.tolist(),
            "trained_positions": trained.tolist(),
            "trained_ids": labels[trained].tolist(),
            "n_image_pad": int((ids == pad_id).sum()),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
            "pixel_mean": float(pixel.mean()),
            "pixel_std": float(pixel.std()),
            "mm_token_type_ids_sum": int(batch["mm_token_type_ids"].sum()) if "mm_token_type_ids" in batch else None,
            "decoded_trained": tok.decode(labels[trained]),
            "decoded_prefix": tok.decode(ids[:12]),
        })
    Path(args.out).write_text(json.dumps({"n": len(samples), "samples": samples}, indent=1))
    print(f"nemo audit dump: {len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
