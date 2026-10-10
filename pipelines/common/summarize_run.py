"""
Summarize one run directory into run_summary.json (stdlib only, runs with any python3).

Inputs (written by run_lib.sh and the hooks):
  timing.json, hparams.env, run_info.txt, gpu.csv, host_mem.csv, train.log (timestamped lines),
  steps.jsonl (HF-Trainer frameworks via the hook) -- otherwise step lines are parsed from train.log
  (NeMo AutoModel prints its own step lines), param_census.json, vision_grad.rank*.json
Step timing: load time = first logged step - run start; steady-state step time = median of the
intervals between consecutive steps from step 3 on; samples/s = GLOBAL_BS / step time; tokens/s uses
the package's per-sample token counts (records/token_lengths.json) over the run's training split.
"""

import json
import re
import statistics
import sys
from pathlib import Path

NEMO_STEP_RE = re.compile(r"\bstep\s+(\d+)\b.*?\bloss\s*[=:]?\s*([0-9]+\.[0-9]+(?:e-?\d+)?)", re.I)


def read_env(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            values[key] = value
    return values


def steps_from_hook(run):
    path = run / "steps.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"step": r["step"], "time": r["time"], "loss": r.get("loss")} for r in rows]


def steps_from_log(run):
    steps, seen = [], set()
    log = run / "train.log"
    if not log.exists():
        return steps
    for line in log.read_text(errors="replace").splitlines():
        stamp, _, text = line.partition(" ")
        match = NEMO_STEP_RE.search(text)
        if match and int(match.group(1)) not in seen:
            seen.add(int(match.group(1)))
            steps.append({"step": int(match.group(1)), "time": float(stamp), "loss": float(match.group(2))})
    return steps


def gpu_stats(run, t0, t1):
    peak, util = {}, {}
    path = run / "gpu.csv"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        try:
            t, idx, mem, ut = line.split(",")
            t, mem, ut = float(t), float(mem), float(ut)
        except ValueError:
            continue
        peak[idx] = max(peak.get(idx, 0.0), mem)
        if t0 is not None and t0 <= t <= t1:
            util.setdefault(idx, []).append(ut)
    return {"peak_mem_mib": peak,
            "max_peak_mem_gib": round(max(peak.values()) / 1024, 2) if peak else None,
            "mean_util_pct_training": {k: round(statistics.mean(v), 1) for k, v in util.items()}}


def host_stats(run):
    path = run / "host_mem.csv"
    used = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                _, total, avail = line.split(",")
                used.append((float(total) - float(avail)) / 1024 / 1024)
            except ValueError:
                continue
    return {"peak_host_used_gib": round(max(used), 2) if used else None}


def mean_tokens(package, split):
    lengths_path = Path(package) / "records" / "token_lengths.json"
    if not lengths_path.exists():
        return None
    lengths = json.loads(lengths_path.read_text())
    ids_file = Path(package) / "records" / f"{split}.ids.txt"
    if split == "audit32":                       # derived subset: same seeded selection as data_lib
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from data_lib import load_split
        ids = [s.id for s in load_split(package, "audit32")]
    elif ids_file.exists():
        ids = ids_file.read_text().split()
    else:
        flat = Path(package) / "records" / f"{split}.flat.jsonl"
        ids = [json.loads(line)["id"] for line in flat.read_text().splitlines() if line.strip()] if flat.exists() else []
    values = [lengths[i]["total"] for i in ids if i in lengths]
    return statistics.mean(values) if values else None


def main(run):
    run = Path(run)
    timing = json.loads((run / "timing.json").read_text()) if (run / "timing.json").exists() else {}
    hp = read_env(run / "hparams.env")
    info = read_env(run / "run_info.txt")
    steps = steps_from_hook(run) or steps_from_log(run)
    steps.sort(key=lambda s: s["time"])

    summary = {"run": run.name, "framework": info.get("framework"), "strategy": hp.get("STRATEGY"),
               "ngpu": hp.get("NGPU"), "global_bs": hp.get("GLOBAL_BS"), "devices": hp.get("CUDA_VISIBLE_DEVICES"),
               "model": Path(info.get("model", "")).name, "rc": timing.get("rc"),
               "steps_logged": len(steps), "wall_s": None, "load_s": None, "step_time_s": None,
               "samples_per_s": None, "tokens_per_s": None,
               "first_loss": steps[0]["loss"] if steps else None,
               "last_loss": steps[-1]["loss"] if steps else None}
    if timing.get("start") and timing.get("end"):
        summary["wall_s"] = round(timing["end"] - timing["start"], 1)
    t0 = t1 = None
    if steps and timing.get("start"):
        t0, t1 = steps[0]["time"], steps[-1]["time"]
        summary["load_s"] = round(t0 - timing["start"], 1)
        intervals = [b["time"] - a["time"] for a, b in zip(steps, steps[1:])]
        steady = intervals[2:] if len(intervals) > 3 else intervals
        if steady:
            step_time = statistics.median(steady)
            summary["step_time_s"] = round(step_time, 3)
            global_bs = int(hp.get("GLOBAL_BS", "32"))
            summary["samples_per_s"] = round(global_bs / step_time, 3)
            tokens = mean_tokens(info.get("package", ""), hp.get("TRAIN_SPLIT", "train"))
            if tokens:
                summary["tokens_per_s"] = round(global_bs * tokens / step_time, 1)
    summary.update(gpu_stats(run, t0, t1))
    summary.update(host_stats(run))
    census = run / "param_census.json"
    if census.exists():
        summary["param_census"] = json.loads(census.read_text())
    probes = sorted(run.glob("vision_grad.rank*.json"))
    if probes:
        summary["vision_output_requires_grad"] = any(json.loads(p.read_text())["output_requires_grad"]
                                                     for p in probes)
    (run / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] {summary['framework']} {summary['model']} {summary['strategy']} x{summary['ngpu']}: "
          f"rc={summary['rc']} steps={summary['steps_logged']} load={summary['load_s']}s "
          f"step={summary['step_time_s']}s tok/s={summary['tokens_per_s']} "
          f"peak={summary.get('max_peak_mem_gib')}GiB host={summary.get('peak_host_used_gib')}GiB "
          f"loss {summary['first_loss']} -> {summary['last_loss']}")


if __name__ == "__main__":
    main(sys.argv[1])
