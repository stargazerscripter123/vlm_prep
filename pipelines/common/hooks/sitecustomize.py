"""
Opt-in instrumentation shared by all 5 frameworks. Active only when VLM_HOOKS=1 and this directory
is on PYTHONPATH (Python imports `sitecustomize` at startup in every process, torchrun workers too).

Nothing is imported eagerly: patches are applied by a post-import hook when the target module is
first imported (so frameworks that must be imported before transformers, e.g. Unsloth, are unaffected).

1. Vision-grad probe (transformers.models.qwen3_vl.modeling_qwen3_vl): wraps Qwen3VLVisionModel.forward;
   the first training-mode call per process writes $VLM_RUN/vision_grad.rank<R>.json with whether the
   frozen vision tower's output requires grad (it must not: no backward through the ViT).
2. Trainer callback (transformers.trainer): appended to every transformers.Trainer (ms-swift, TRL,
   Unsloth and Axolotl all subclass it):
   - steps.jsonl      : {"step", "time", "loss", ...} per logging step (rank 0)
   - param_census.json: trainable parameters at train begin (names, numel, dtypes, forbidden modules)
   - audit_dump.json  : if VLM_AUDIT_DUMP=1, the first VLM_AUDIT_N training batches exactly as the
                        trainer's own dataloader/collator produce them (ids, trained label positions,
                        image grid, pixel stats, mm_token_type_ids) -- run single-GPU on audit32.
"""

import functools
import importlib.abc
import importlib.util
import json
import os
import sys
import time

if os.environ.get("VLM_HOOKS") == "1":
    _RUN = os.environ.get("VLM_RUN", ".")
    _RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    _FORBIDDEN = ("visual", "merger", "lm_head", "embed_tokens")

    def _write(name, payload):
        try:
            with open(os.path.join(_RUN, name), "w") as f:
                json.dump(payload, f, indent=1, default=str)
        except Exception as exc:  # instrumentation must never break training
            print(f"[vlm-hooks] cannot write {name}: {exc}", file=sys.stderr)

    def _first_tensor(value):
        import torch
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, (list, tuple)):
            for item in value:
                found = _first_tensor(item)
                if found is not None:
                    return found
        for attr in ("pooler_output", "last_hidden_state"):
            if hasattr(value, attr):
                return _first_tensor(getattr(value, attr))
        return None

    def _patch_qwen3_vl(module):
        cls = getattr(module, "Qwen3VLVisionModel", None)
        if cls is None or getattr(cls, "_vlm_probe", False):
            return
        original = cls.forward
        state = {"done": False}

        @functools.wraps(original)        # keep the signature: frameworks inspect it
        def forward(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            if not state["done"] and self.training:
                state["done"] = True
                tensor = _first_tensor(out)
                _write(f"vision_grad.rank{_RANK}.json", {
                    "output_requires_grad": bool(tensor is not None and tensor.requires_grad),
                    "output_has_grad_fn": bool(tensor is not None and tensor.grad_fn is not None),
                    "any_vision_param_trainable": any(p.requires_grad for p in self.parameters()),
                    "time": time.time()})
            return out

        cls.forward = forward
        cls._vlm_probe = True

    def _make_callback(trainer_module):
        TrainerCallback = trainer_module.TrainerCallback

        class VlmCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, train_dataloader=None,
                               processing_class=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                if model is not None:
                    try:
                        _write("param_census.json", _census(model))
                    except Exception as exc:
                        print(f"[vlm-hooks] census failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                if os.environ.get("VLM_AUDIT_DUMP") == "1" and train_dataloader is not None:
                    try:      # a failing dump must not break training; the audit then fails on the missing file
                        _write("audit_dump.json", _dump(train_dataloader, processing_class,
                                                        int(os.environ.get("VLM_AUDIT_N", "32"))))
                    except Exception as exc:
                        print(f"[vlm-hooks] audit dump failed: {type(exc).__name__}: {exc}", file=sys.stderr)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.is_world_process_zero and logs and "loss" in logs:
                    with open(os.path.join(_RUN, "steps.jsonl"), "a") as f:
                        f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs},
                                           default=float) + "\n")

        return VlmCallback

    def _census(model):
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        numel = sum(getattr(p, "ds_numel", p.numel()) for _, p in trainable)
        return {
            "trainable_tensors": len(trainable),
            "trainable_params": int(numel),
            "lora_modules": len({n.rsplit(".lora_", 1)[0] for n, _ in trainable if ".lora_" in n}),
            "dtypes": sorted({str(p.dtype) for _, p in trainable}),
            "forbidden": [n for n, _ in trainable if any(f in n for f in _FORBIDDEN)][:20],
            "examples": [n for n, _ in trainable[:4]],
        }

    def _tokenizer_of(processing_class):
        return getattr(processing_class, "tokenizer", processing_class)

    def _dump(dataloader, processing_class, limit):
        tok = _tokenizer_of(processing_class)
        pad_id = tok.convert_tokens_to_ids("<|image_pad|>") if tok is not None else None
        samples = []
        for batch in dataloader:
            ids = batch["input_ids"]
            labels = batch.get("labels")
            for row in range(ids.shape[0]):
                mask = batch["attention_mask"][row].bool() if "attention_mask" in batch else None
                row_ids = ids[row][mask] if mask is not None else ids[row]
                row_labels = labels[row][mask] if (labels is not None and mask is not None) else labels[row]
                trained = (row_labels != -100).nonzero().flatten().tolist()
                pixel = batch.get("pixel_values")
                entry = {
                    "input_ids": row_ids.tolist(),
                    "trained_positions": trained,
                    "trained_ids": row_labels[trained].tolist(),
                    "n_image_pad": int((row_ids == pad_id).sum()) if pad_id is not None else None,
                    "image_grid_thw": batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else None,
                    "pixel_mean": float(pixel.float().mean()) if pixel is not None and ids.shape[0] == 1 else None,
                    "pixel_std": float(pixel.float().std()) if pixel is not None and ids.shape[0] == 1 else None,
                    "mm_token_type_ids_sum": (int(batch["mm_token_type_ids"][row].sum())
                                              if "mm_token_type_ids" in batch else None),
                    "decoded_trained": tok.decode(row_labels[trained]) if tok is not None else None,
                    "decoded_prefix": tok.decode(row_ids[:12]) if tok is not None else None,
                }
                samples.append(entry)
            if len(samples) >= limit:
                break
        return {"n": len(samples), "samples": samples[:limit]}

    def _patch_trainer(module):
        Trainer = getattr(module, "Trainer", None)
        if Trainer is None or getattr(Trainer, "_vlm_patched", False):
            return
        callback_cls = _make_callback(module)
        original_init = Trainer.__init__

        @functools.wraps(original_init)   # ms-swift picks processing_class vs tokenizer from this signature
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                self.add_callback(callback_cls())
            except Exception as exc:
                print(f"[vlm-hooks] add_callback failed: {exc}", file=sys.stderr)

        Trainer.__init__ = __init__
        Trainer._vlm_patched = True

    _HOOKS = {
        "transformers.models.qwen3_vl.modeling_qwen3_vl": _patch_qwen3_vl,
        "transformers.trainer": _patch_trainer,
    }

    class _PostImportFinder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = set()

        def find_spec(self, fullname, path, target=None):
            if fullname not in _HOOKS or fullname in self._busy:
                return None
            self._busy.add(fullname)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy.discard(fullname)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            loader, hook = spec.loader, _HOOKS[fullname]
            original_exec = loader.exec_module

            def exec_module(module):
                original_exec(module)
                try:
                    hook(module)
                except Exception as exc:
                    print(f"[vlm-hooks] patch {fullname} failed: {exc}", file=sys.stderr)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PostImportFinder())
