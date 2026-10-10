"""
Guard for hosts where direct GPU-to-GPU copies (cudaMemcpyPeer) silently corrupt data.

Found on popos-6000b (4x RTX 6000 Ada, PCIe): torch reports peer access, but tensor.to("cuda:j") from
another GPU returns garbage (~100% of elements wrong). NCCL collectives are unaffected (verified exactly
by nccl_test.py), so DDP / ZeRO-3 / FSDP training is fine; only single-process model splitting
(device_map="auto": reference loss / evaluation of the 32B model on 48 GB cards) is hit.

ensure_p2p_safe(): checks a cross-device copy; if it is corrupted, routes every cuda->other-cuda
Tensor.to() through host memory (correct; hidden states between pipeline stages are small).
"""

import torch

_installed = False


def p2p_copies_ok():
    if torch.cuda.device_count() < 2:
        return True
    src = torch.randn(1024 * 1024, device="cuda:0")
    reference = src.cpu()
    return all(torch.equal(src.to(f"cuda:{j}").cpu(), reference) for j in range(1, torch.cuda.device_count()))


def install():
    global _installed
    if _installed:
        return
    original_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        if self.is_cuda:
            try:   # _parse_to knows device/dtype/non_blocking/memory_format, not Tensor.to's `copy`
                device = torch._C._nn._parse_to(*args, **{k: v for k, v in kwargs.items() if k != "copy"})[0]
            except (RuntimeError, TypeError):
                device = None
            if device is not None and device.type == "cuda" and device.index is not None \
                    and device.index != self.device.index:
                return original_to(original_to(self, "cpu"), *args, **kwargs)
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = to
    _installed = True


def ensure_p2p_safe(verbose=True):
    """Install the host-staged copy path when direct device-to-device copies are corrupted."""
    if p2p_copies_ok():
        return False
    install()
    if not p2p_copies_ok():
        raise RuntimeError("cross-GPU copies are still corrupted after enabling host staging")
    if verbose:
        print("[p2p_safe] direct GPU-to-GPU copies are corrupted on this host -> staging through CPU")
    return True
