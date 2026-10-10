"""
Reference encoding + base-model loss for the cross-framework audit (run in the eval env: TRL env).

For every sample of a split (default audit32) the HF processor + Qwen3-VL chat template produce the
reference input_ids / image grid / pixel stats; the trained tokens are the assistant answer plus
<|im_end|> (convention "eos") or plus <|im_end|> and the following "\\n" (convention "eos_nl").
With --loss the BASE model (LoRA B=0 at step 1, so this equals every framework's first-step loss)
gives token-weighted mean CE over the split for both conventions.

  python reference_encode.py --pkg PKG --model MODEL --out runs/reference/audit32_<model>.json [--loss]
  python reference_encode.py --pkg PKG --model MODEL --out drift_<fw>.json --loss --limit 3   # env drift
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_lib import load_split  # noqa: E402

MIN_PIXELS = int(os.environ.get("MIN_PIXELS", "65536"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "1048576"))


def parse_args():
    parser = argparse.ArgumentParser(description="Reference encoding / loss for the audit")
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="audit32")
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_processor(model_dir):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    for attr, value in (("min_pixels", MIN_PIXELS), ("max_pixels", MAX_PIXELS)):
        try:  # read-only property (derived from size) in some transformers versions
            setattr(image_processor, attr, value)
        except AttributeError:
            pass
    return processor


def messages_for(sample):
    return [{"role": "user", "content": [{"type": "image", "image": sample.image},
                                         {"type": "text", "text": sample.prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}]


def encode(processor, sample):
    batch = processor.apply_chat_template(messages_for(sample), tokenize=True, return_dict=True,
                                          return_tensors="pt", add_generation_prompt=False)
    tok = processor.tokenizer
    ids = batch["input_ids"][0].tolist()
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    header = tok.encode("assistant\n", add_special_tokens=False)
    starts = [i for i, t in enumerate(ids) if t == im_start and ids[i + 1:i + 1 + len(header)] == header]
    begin = starts[-1] + 1 + len(header)
    end = ids.index(im_end, begin)                      # inclusive <|im_end|>
    positions_eos = list(range(begin, end + 1))
    positions_nl = positions_eos + ([end + 1] if end + 1 < len(ids) else [])
    pixel = batch["pixel_values"].float()
    return batch, {
        "id": sample.id,
        "input_ids": ids,
        "trained_positions": {"eos": positions_eos, "eos_nl": positions_nl},
        "trained_ids": {"eos": [ids[p] for p in positions_eos], "eos_nl": [ids[p] for p in positions_nl]},
        "n_image_pad": ids.count(tok.convert_tokens_to_ids("<|image_pad|>")),
        "image_grid_thw": batch["image_grid_thw"].tolist(),
        "pixel_mean": float(pixel.mean()),
        "pixel_std": float(pixel.std()),
        "mm_token_type_ids_sum": int(batch["mm_token_type_ids"].sum()) if "mm_token_type_ids" in batch else None,
        "has_system": "<|im_start|>system" in tok.decode(ids[:8]),
    }


def main():
    args = parse_args()
    samples = load_split(args.pkg, args.split)[: args.limit]
    processor = load_processor(args.model)
    model = None
    if args.loss:
        from p2p_safe import ensure_p2p_safe
        from transformers import AutoModelForImageTextToText

        ensure_p2p_safe()          # device_map="auto" copies hidden states between GPUs
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map="auto", attn_implementation="sdpa")
        model.eval()

    entries, sums = [], {"eos": [0.0, 0], "eos_nl": [0.0, 0]}
    for sample in samples:
        batch, entry = encode(processor, sample)
        if model is not None:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**inputs).logits[0].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            ids = batch["input_ids"][0].to(log_probs.device)
            entry["ce"] = {}
            for convention, positions in entry["trained_positions"].items():
                pos = torch.tensor(positions, device=log_probs.device)
                nll = -log_probs[pos - 1, ids[pos]]
                entry["ce"][convention] = {"sum": float(nll.sum()), "n": len(positions)}
                sums[convention][0] += float(nll.sum())
                sums[convention][1] += len(positions)
        entries.append(entry)

    out = {"package": str(Path(args.pkg).resolve()), "model": str(Path(args.model).resolve()),
           "split": args.split, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
           "transformers": __import__("transformers").__version__, "samples": entries}
    if model is not None:
        out["loss"] = {c: s / n for c, (s, n) in sums.items() if n}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"reference: {len(entries)} samples -> {args.out}"
          + (f"  loss eos={out['loss']['eos']:.4f} eos_nl={out['loss']['eos_nl']:.4f}" if model is not None else ""))


if __name__ == "__main__":
    main()
