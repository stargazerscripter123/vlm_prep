"""
Axolotl plugin (listed under `plugins:` in qwen3vl_lora.yaml.tmpl; train.sh puts this dir on PYTHONPATH).

Axolotl 0.20.0 bug: rows of cfg.role_boundaries arrive as DictDefault objects, whose attribute access
returns None for missing names. processing_strategies._resolve_role_boundary_override checks
`hasattr(spec, "model_dump")` (True for DictDefault) and calls spec.model_dump() -> None() -> TypeError.
Converting the rows to plain dicts before the trainer/collator is built takes its `dict(spec)` path.
"""

from axolotl.integrations.base import BasePlugin


class VlmFixesPlugin(BasePlugin):
    def pre_model_load(self, cfg):
        if cfg.role_boundaries:
            cfg.role_boundaries = [dict(spec) for spec in cfg.role_boundaries]
            print(f"[vlm_fixes] role_boundaries as plain dicts: {[b['role'] for b in cfg.role_boundaries]}")

    def post_trainer_create(self, cfg, trainer):
        # transformers sets Qwen3-VL `accepts_loss_kwargs = False` (a copy of the Gemma3 grad-acc fix), so
        # Trainer averages the loss per micro-batch (= per SAMPLE at micro-batch 1). Qwen3-VL's forward
        # passes **kwargs to self.loss_function, which normalizes by num_items_in_batch correctly, so turn
        # it on: loss over the trained tokens of the whole global batch (token-weighted), like the other
        # pipelines. Verified by the step-1 loss audit.
        trainer.model_accepts_loss_kwargs = True
        print("[vlm_fixes] token-weighted loss: trainer.model_accepts_loss_kwargs = True")
