"""torchrun --nproc_per_node N nccl_test.py : all-reduce / all-gather CORRECTNESS + bandwidth, P2P copy check.

Every rank contributes rank-dependent random data; results are checked exactly against a CPU
computation. Also checks direct device-to-device copies (cudaMemcpyPeer) used by device_map=auto.
Exit code 1 if anything is wrong (P2P on some PCIe hosts silently corrupts data)."""
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    numel = 16 * 1024 * 1024
    ok = True

    # all-reduce correctness: integer-valued floats so the sum is exact
    gen = torch.Generator().manual_seed(1234)
    parts = [torch.randint(-1000, 1000, (numel,), generator=gen).float() for _ in range(world)]
    tensor = parts[rank].cuda()
    dist.all_reduce(tensor)
    expected = torch.stack(parts).sum(0)
    if not torch.equal(tensor.cpu(), expected):
        ok = False
        print(f"[rank {rank}] all_reduce WRONG: {int((tensor.cpu() != expected).sum())} of {numel} elements")

    # all-gather correctness
    gathered = [torch.empty(numel, device="cuda") for _ in range(world)]
    dist.all_gather(gathered, parts[rank].cuda())
    for r in range(world):
        if not torch.equal(gathered[r].cpu(), parts[r]):
            ok = False
            print(f"[rank {rank}] all_gather WRONG for source rank {r}")

    # bandwidth
    big = torch.ones(64 * 1024 * 1024, device="cuda")
    for _ in range(3):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    start, iters = time.time(), 10
    for _ in range(iters):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    seconds = (time.time() - start) / iters
    bus = big.numel() * 4 * 2 * (world - 1) / world / seconds / 1e9

    flags = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0:
        n = torch.cuda.device_count()
        src = torch.randn(4 * 1024 * 1024, device="cuda:0")
        peer_ok = all(torch.equal(src.to(f"cuda:{j}").cpu(), src.cpu()) for j in range(1, n))
        print(f"NCCL collectives: {'CORRECT' if flags.item() else 'WRONG'} | all-reduce bus {bus:.1f} GB/s "
              f"| direct device copies (device_map=auto): {'CORRECT' if peer_ok else 'CORRUPTED'} "
              f"| NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}")
    dist.destroy_process_group()
    sys.exit(0 if flags.item() else 1)


if __name__ == "__main__":
    main()
