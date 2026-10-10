"""
Correctness checks for a run directory (stdlib only). Exit code 1 on any failure.

  python compare_audit.py --reference runs/reference/audit32_<model>_<pkg>.json --run <run_dir>   # full audit
  python compare_audit.py --run <run_dir> --post-train     # after any training run: parameters + frozen ViT

Full audit, per sample (matched by trained answer tokens + image grid + pixel mean):
  - input ids identical to the reference (NeMo-style shifted inputs may lack the final token)
  - trained tokens == answer + <|im_end|> (convention "eos") or + "\n" ("eos_nl"); one convention per run
  - same number of <|image_pad|> tokens and identical image_grid_thw; pixel mean/std within 0.02
  - no system prompt
  - step-1 loss (steps.jsonl or parsed log) within --loss-tol of the reference loss for the run's convention
Run-level checks (both modes):
  - trainable parameters: LoRA on exactly q/k/v/o/gate/up/down of every text layer (expected count and
    module count computed from the model's config.json), nothing under visual / merger / lm_head /
    embeddings -- from param_census.json (hook, transformers.Trainer frameworks) and from the saved
    adapter file (all frameworks). Not being able to compute the expectation is a failure.
  - vision_grad.rank*.json: the frozen vision tower's output must not require grad
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_run import steps_from_hook, steps_from_log  # noqa: E402

FORBIDDEN = ("visual", "merger", "lm_head", "embed_tokens")


def read_env(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            values[key] = value
    return values


def expected_census(run):
    """(trainable params, LoRA modules) for LoRA r on q/k/v/o/gate/up/down of every text layer, from the
    run's model config (any Qwen3-VL size). Raises ValueError when it cannot be computed."""
    model = read_env(run / "run_info.txt").get("model")
    rank = int(read_env(run / "hparams.env").get("LORA_R", "0") or 0)
    config_path = Path(model or "") / "config.json"
    if not model or not config_path.exists():
        raise ValueError(f"model config not readable ({config_path})")
    if not rank:
        raise ValueError("LORA_R missing from hparams.env")
    text = json.loads(config_path.read_text()).get("text_config", {})
    try:
        hidden, inter, layers = text["hidden_size"], text["intermediate_size"], text["num_hidden_layers"]
        heads, kv_heads, head_dim = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
    except KeyError as exc:
        raise ValueError(f"text_config lacks {exc}") from exc
    q_out, kv_out = heads * head_dim, kv_heads * head_dim
    per_layer = ((hidden + q_out) + 2 * (hidden + kv_out) + (q_out + hidden)     # q, k, v, o
                 + 2 * (hidden + inter) + (inter + hidden))                       # gate, up, down
    return rank * per_layer * layers, 7 * layers


def adapter_census(path):
    """Census of a saved PEFT adapter from its safetensors header (no tensor data is read)."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    params = 0
    for meta in tensors.values():
        count = 1
        for dim in meta["shape"]:
            count *= dim
        params += count
    return {
        "trainable_params": params,
        "lora_modules": len({k.rsplit(".lora_", 1)[0] for k in tensors if ".lora_" in k}),
        "forbidden": [k for k in tensors if any(f in k for f in FORBIDDEN)][:20],
        "not_lora": [k for k in tensors if ".lora_" not in k][:20],
    }


def check_run(run, failures, notes, require_adapter):
    """Trainable-parameter and frozen-ViT checks shared by the audit and the post-train check."""
    try:
        want_params, want_modules = expected_census(run)
    except ValueError as exc:
        failures.append(f"expected LoRA parameter count cannot be computed: {exc}")
        want_params = want_modules = None

    sources = {}
    if (run / "param_census.json").exists():
        sources["hook census"] = json.loads((run / "param_census.json").read_text())
    adapter = run / "adapter" / "adapter_model.safetensors"
    if adapter.exists():
        sources["saved adapter"] = adapter_census(adapter)
    elif require_adapter:
        failures.append("no saved adapter (adapter/adapter_model.safetensors)")
    if not sources:
        failures.append("no parameter census available (neither the hook census nor a saved adapter)")
    for name, census in sources.items():
        if census["forbidden"]:
            failures.append(f"{name}: trainable parameters under frozen modules: {census['forbidden'][:3]}")
        if census.get("not_lora"):
            failures.append(f"{name}: non-LoRA tensors in the adapter: {census['not_lora'][:3]}")
        if want_params is not None:
            if census["trainable_params"] != want_params:
                failures.append(f"{name}: trainable params {census['trainable_params']:,} != expected {want_params:,}")
            if census["lora_modules"] != want_modules:
                failures.append(f"{name}: LoRA modules {census['lora_modules']} != expected {want_modules}")
        notes.append(f"{name}: {census['trainable_params']:,} params, {census['lora_modules']} LoRA modules"
                     + (f", dtypes {census['dtypes']}" if "dtypes" in census else ""))

    probes = sorted(run.glob("vision_grad.rank*.json"))
    if probes:
        grads = [json.loads(p.read_text()) for p in probes]
        if any(g["output_requires_grad"] for g in grads):
            failures.append("vision tower output requires grad -> backward runs through the frozen ViT")
        if any(g["any_vision_param_trainable"] for g in grads):
            failures.append("a vision tower parameter is trainable")
        notes.append(f"vision probe: {len(grads)} rank(s), output_requires_grad="
                     f"{[g['output_requires_grad'] for g in grads]}")
    else:
        failures.append("vision-grad probe did not fire (cannot show that the frozen ViT gets no backward)")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit / post-train correctness checks of a run")
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", default=None, help="reference json (full audit)")
    parser.add_argument("--post-train", action="store_true",
                        help="only the run-level checks (parameters, frozen ViT); no reference needed")
    parser.add_argument("--dump", default=None, help="default: <run>/audit_dump.json")
    parser.add_argument("--loss-tol", type=float, default=0.03)
    args = parser.parse_args()
    if not args.post_train and not args.reference:
        parser.error("--reference is required unless --post-train")
    return args


def finish(failures, notes, label):
    for note in notes:
        print(f"  - {note}")
    if failures:
        for failure in failures[:40]:
            print(f"FAIL: {failure}")
        print(f"{label} FAILED: {len(failures)} problem(s)")
        sys.exit(1)
    print(f"{label} OK")


def match(reference, dumped):
    """Map each dumped sample to a reference sample (answers can repeat, e.g. negatives)."""
    used, pairs = set(), []
    for d in dumped:
        best = None
        for i, r in enumerate(reference):
            if i in used:
                continue
            if d["trained_ids"] not in (r["trained_ids"]["eos"], r["trained_ids"]["eos_nl"]):
                continue
            score = abs((d.get("pixel_mean") or 0) - r["pixel_mean"]) + (0 if d.get("image_grid_thw") == r["image_grid_thw"] else 10)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((d, reference[best[1]]))
        else:
            pairs.append((d, None))
    return pairs


def main():
    args = parse_args()
    run = Path(args.run)
    failures, notes = [], []
    if args.post_train:
        check_run(run, failures, notes, require_adapter=True)
        finish(failures, notes, "POST-TRAIN CHECK")
        return
    reference = json.loads(Path(args.reference).read_text())

    dump_path = Path(args.dump) if args.dump else run / "audit_dump.json"
    conventions = set()
    if dump_path.exists():
        dumped = json.loads(dump_path.read_text())["samples"]
        pairs = match(reference["samples"], dumped)
        for d, r in pairs:
            if r is None:
                failures.append(f"sample with trained text {d.get('decoded_trained', '')[:60]!r} has no reference match")
                continue
            rid = r["id"]
            convention = "eos" if d["trained_ids"] == r["trained_ids"]["eos"] else "eos_nl"
            conventions.add(convention)
            ids, ref_ids = d["input_ids"], r["input_ids"]
            if ids != ref_ids and not (len(ids) == len(ref_ids) - 1 and ids == ref_ids[:-1]):
                diff = next((i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b), min(len(ids), len(ref_ids)))
                failures.append(f"{rid}: input_ids differ at {diff} (len {len(ids)} vs {len(ref_ids)}); "
                                f"prefix {d.get('decoded_prefix')!r}")
            if d.get("n_image_pad") is not None and d["n_image_pad"] != r["n_image_pad"]:
                failures.append(f"{rid}: image pads {d['n_image_pad']} != {r['n_image_pad']}")
            if d.get("image_grid_thw") is not None and d["image_grid_thw"] != r["image_grid_thw"]:
                failures.append(f"{rid}: image_grid_thw {d['image_grid_thw']} != {r['image_grid_thw']}")
            for key in ("pixel_mean", "pixel_std"):
                if d.get(key) is not None and abs(d[key] - r[key]) > 0.02:
                    failures.append(f"{rid}: {key} {d[key]:.4f} vs {r[key]:.4f}")
            if r["has_system"]:
                failures.append(f"{rid}: reference unexpectedly has a system prompt")
        notes.append(f"audit dump: {len(pairs)} samples matched, convention(s) {sorted(conventions)}")
        if len(dumped) != len(reference["samples"]):
            failures.append(f"audit dump has {len(dumped)} samples, the reference {len(reference['samples'])}")
        if len(conventions) > 1:
            failures.append(f"mixed trained-token conventions within one run: {sorted(conventions)}")
    else:
        failures.append(f"missing {dump_path.name}: the framework's batches were not captured")

    check_run(run, failures, notes, require_adapter=False)

    steps = steps_from_hook(run) or steps_from_log(run)
    if not steps:
        failures.append("no logged training step: step-1 loss cannot be checked")
    elif "loss" in reference:
        first = steps[0]["loss"]
        convention = conventions.pop() if len(conventions) == 1 else None
        candidates = [convention] if convention else ["eos", "eos_nl"]
        errors = {c: abs(first - reference["loss"][c]) / reference["loss"][c] for c in candidates}
        best = min(errors, key=errors.get)
        notes.append(f"step-1 loss {first:.4f} vs reference {reference['loss'][best]:.4f} ({best}): "
                     f"{100 * errors[best]:.2f}%")
        if errors[best] > args.loss_tol:
            failures.append(f"step-1 loss off by {100 * errors[best]:.1f}% (> {100 * args.loss_tol:.0f}%)")

    finish(failures, notes, "AUDIT")


if __name__ == "__main__":
    main()
