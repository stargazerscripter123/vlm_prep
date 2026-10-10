"""
ms-swift plugin (loaded with --external_plugins): stop backpropagation through the frozen vision tower.

With gradient checkpointing, ms-swift 4.5.3 calls model.enable_input_require_grads(). In transformers 5.x
that registers a forward hook on the input embeddings of EVERY PreTrainedModel inside the model --
including the vision tower (ms-swift maps visual.get_input_embeddings to its patch embedding). The
following vision_tower.disable_input_require_grads() only removes hooks stored on the vision tower
itself, so the ViT output keeps requiring grad and every step runs a useless backward pass through the
frozen ViT (no vision parameter is trainable; the LoRA gradients are unchanged). This removes exactly
those hooks right after ms-swift prepared gradient checkpointing; the language-model hook stays.
Found by the cross-framework audit (vision_grad probe), see pipelines/common/compare_audit.py.
"""

from swift.trainers import mixin

_original = mixin.SwiftMixin._prepare_gradient_checkpointing


def _prepare_gradient_checkpointing(self, model):
    _original(self, model)
    removed = 0
    for name, module in model.named_modules():
        if not name.endswith("visual") or not hasattr(module, "get_input_embeddings"):
            continue
        try:
            embeddings = module.get_input_embeddings()
        except Exception:
            continue
        if embeddings is None:
            continue
        for hook_id, hook in list(embeddings._forward_hooks.items()):
            if getattr(hook, "__name__", "") == "make_inputs_require_grads":
                del embeddings._forward_hooks[hook_id]
                removed += 1
    print(f"[no_vit_backward] removed {removed} input-require-grad hook(s) from the vision tower")


mixin.SwiftMixin._prepare_gradient_checkpointing = _prepare_gradient_checkpointing
