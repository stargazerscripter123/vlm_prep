"""
Unified inference for every framework's LoRA adapter (and zero-shot baselines). Run in the TRL env.

  python predict.py --model DIR [--adapter DIR] --pkg PKG --subset n100 --out OUT_DIR [--device-map auto]
  torchrun --nproc_per_node N predict.py ...      # data-parallel when the model fits on one GPU

Adapter loading is strict (PEFT itself only warns): the adapter's tensor names must equal exactly the
LoRA parameters PEFT creates on this model; LoRA module and parameter counts must match the model;
every lora_B must be non-zero; logits on a probe sample must change vs. the base model. LoRA is never
merged. Decoding is explicitly greedy (the checkpoint's generation_config samples by default),
batch size 1, Qwen3-VL template without a system prompt, max_new_tokens 1024.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "--device-map" not in sys.argv:
    # Data-parallel eval: each rank sees ONLY its own GPU (set before torch initializes CUDA), so no
    # library code can make a cross-GPU copy -- direct peer copies silently corrupt data on some hosts
    # (popos-6000b; see pipelines/common/p2p_safe.py).
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(64))).split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible[int(os.environ["LOCAL_RANK"])]

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import eval_subset, load_split  # noqa: E402

MIN_PIXELS = int(os.environ.get("MIN_PIXELS", "65536"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "1048576"))
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Qwen3-VL grounding inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--split", default="person_test")
    parser.add_argument("--subset", default="n100", help="n100 | n50 | full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device-map", default=None, help="'auto' to split one model over all GPUs")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "32")))
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


def build_inputs(processor, sample, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": sample.image},
                                             {"type": "text", "text": sample.prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def expected_lora(model, r):
    """Expected LoRA module count and parameter count for r on LORA_TARGETS of the language model."""
    modules, params = 0, 0
    for name, module in model.named_modules():
        if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in LORA_TARGETS \
                and isinstance(module, torch.nn.Linear):
            modules += 1
            params += r * (module.in_features + module.out_features)
    return modules, params


def load_adapter_strict(model, adapter_dir, processor, probe, r):
    from peft import PeftModel
    from safetensors.torch import load_file

    want_modules, want_params = expected_lora(model, r)
    device = next(model.parameters()).device
    probe_inputs = build_inputs(processor, probe, device)
    with torch.no_grad():
        base_logits = model(**probe_inputs).logits[0, -1].float().cpu()

    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False, torch_device="cpu")
    created = {n.replace(".default", ""): p for n, p in peft_model.named_parameters() if ".lora_" in n}
    saved_names = set(saved)
    missing = sorted(set(created) - saved_names)
    unexpected = sorted(saved_names - set(created))
    problems = []
    if missing:
        problems.append(f"{len(missing)} LoRA tensors missing from the adapter, e.g. {missing[:2]}")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected adapter tensors, e.g. {unexpected[:2]}")
    lora_modules = len({n.rsplit(".lora_", 1)[0] for n in created})
    lora_params = sum(p.numel() for p in created.values())
    if lora_modules != want_modules or lora_params != want_params:
        problems.append(f"LoRA modules/params {lora_modules}/{lora_params:,} != expected {want_modules}/{want_params:,}")
    zero_b = [n for n, p in created.items() if ".lora_B" in n and float(p.abs().max()) == 0.0]
    if zero_b:
        problems.append(f"{len(zero_b)} lora_B tensors are all zero (adapter never trained?)")
    with torch.no_grad():
        adapted_logits = peft_model(**probe_inputs).logits[0, -1].float().cpu()
    delta = float((adapted_logits - base_logits).abs().max())
    if delta < 1e-3:
        problems.append(f"adapter does not change the logits (max |delta| {delta:.2e})")
    if problems:
        raise SystemExit("ADAPTER CHECK FAILED:\n  " + "\n  ".join(problems))
    dtypes = sorted({str(t.dtype) for t in saved.values()})
    return peft_model, {"lora_modules": lora_modules, "lora_params": lora_params,
                        "saved_dtypes": dtypes, "probe_logit_delta": delta}


def main():
    args = parse_args()
    from transformers import AutoModelForImageTextToText

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.device_map == "auto":
        from p2p_safe import ensure_p2p_safe

        ensure_p2p_safe()          # one model split over GPUs: guard against corrupted peer copies
        device_map = "auto"
    else:
        device_map = {"": 0}          # the only visible GPU of this rank

    samples = eval_subset(load_split(args.pkg, args.split), args.subset)
    mine = samples[rank::world]
    processor = load_processor(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map=device_map, attn_implementation="sdpa")
    model.eval()
    adapter_info = None
    if args.adapter:
        model, adapter_info = load_adapter_strict(model, args.adapter, processor, samples[0], args.lora_r)

    generation = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                  "max_new_tokens": args.max_new_tokens}
    eos_ids = {processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id}
    device = next(model.parameters()).device
    done, started = 0, time.time()
    with open(out / f"preds.rank{rank}.jsonl", "w") as f:        # one line per sample, flushed immediately
        for sample in mine:
            inputs = build_inputs(processor, sample, device)
            with torch.no_grad():
                output = model.generate(**inputs, **generation)
            new = output[0, inputs["input_ids"].shape[1]:].tolist()
            truncated = len(new) >= args.max_new_tokens and not (set(new[-1:]) & eos_ids)
            f.write(json.dumps({"id": sample.id, "text": processor.decode(new, skip_special_tokens=True),
                                "new_tokens": len(new), "truncated": truncated}) + "\n")
            f.flush()
            done += 1
            if rank == 0 and (done % 25 == 0 or done == len(mine)):
                rate = (time.time() - started) / done
                print(f"[rank 0] {done}/{len(mine)} samples, {rate:.1f} s/sample", flush=True)
    if rank == 0:
        info = {"model": str(Path(args.model).resolve()), "adapter": args.adapter, "subset": args.subset,
                "split": args.split, "n_samples": len(samples), "world": world,
                "generation": generation, "resolved_generation_config": model.generation_config.to_dict(),
                "adapter_check": adapter_info, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                "rank0_seconds": round(time.time() - started, 1),
                "transformers": __import__("transformers").__version__}
        (out / "predict_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"[rank {rank}] {done} predictions -> {out}")


if __name__ == "__main__":
    main()
