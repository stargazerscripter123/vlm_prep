# vlm_prep：Qwen3-VL person 检测框 LoRA 微调

用 COCO 的 person 标注微调 Qwen3-VL（4B / 32B），让模型输出检测框。同一份数据、同一组超参，分别用 5 个框架训练
（ms-swift、TRL + PEFT、NeMo AutoModel、Axolotl、Unsloth），再用同一个脚本评测，比较效果、速度和显存。

| 路径 | 内容 | 进 git |
|---|---|---|
| `scripts/` | 下载 COCO、生成并校验数据包、下载模型 | 是 |
| `pipelines/` | 训练、评测、批量循环。选项、输出文件、各框架的修正见 [pipelines/README.md](pipelines/README.md) | 是 |
| `environment.yml` | 本机跑 `scripts/` 用的 conda 环境（H200 上不需要） | 是 |
| `data/`、`models/` | 数据包、模型权重 | 否 |
| `runs/`、`.cache/` | 训练输出、缓存 | 否 |

入口只有两个脚本：

```bash
bash pipelines/run.sh setup                                              # 每台机器一次：env、模型、数据、主机检查
bash pipelines/run.sh train --fw <框架|all> --model <4b|32b> --gpus <N>   # 正确性检查 → 训练 → 评测
bash pipelines/run.sh status                                             # 汇总表
bash pipelines/sweep.sh                                                  # 模型 × 卡数 × 框架，逐个训练 + 评测
```

## 在 H200 机器上从头跑

所有命令都在 `vlm_prep/` 目录下执行。

### 0. 机器要满足的条件

- **系统**：Linux，`nvidia-smi` 能用，NVIDIA 驱动 ≥ 560；有 `bash`（≥ 4.4）、`python3`、`curl`。
- **网络**：能访问 conda-forge、PyPI、download.pytorch.org、huggingface.co。需要镜像时先 `export HF_ENDPOINT=...`。
- **磁盘**：留 150GB 以上。32B 约 62GB，4B 约 9GB，5 个 conda env 共约 35GB，缓存约 12GB，其余是训练输出。
- **路径**：项目目录的路径里不能带 `Dropbox`（脚本会拒绝写入）。
- **`/dev/shm`**：建议 4GB 以上。更小时脚本会自动关掉 DataLoader worker，训练结果不变，数据加载变慢。

### 1. 把代码和数据放到机器上

`data/` 和 `models/` 不在 git 里，所以有两种带法。

**方式 A：打包（推荐）。** 包里有代码和数据包，约 1.1GB，不含模型。

```bash
# 本机
bash pipelines/tools/pack.sh                 # 生成 vlm_prep_code_data_<日期>.tar
scp vlm_prep_code_data_*.tar <机器>:/workspace/

# H200 机器
cd /workspace && tar xf vlm_prep_code_data_*.tar && cd vlm_prep
```

**方式 B：`git clone`。** 这样机器上没有数据包，二选一：

- 把本机的 `data/package/vlm_sft_person_261009/`（1.1GB）拷到机器上同样的位置；
- 或者什么都不拷，第 2 步会从 COCO 官方服务器重新下载并生成数据包。这条自动生成的路径没有在新机器上完整验证过。

模型两种方式都不用带，第 2 步会自动下载。想省掉 32B 的下载，可以把 `models/Qwen3-VL-32B-Instruct/` 自己拷过去，
第 2 步只做校验。

### 2. 准备环境（每台机器一次）

```bash
bash pipelines/run.sh setup
```

依次做这些事，缺什么补什么，可以重复运行：

- 建 5 个 conda env（`vlm<框架>_NV_py312`），按驱动版本自动选 PyTorch 的 CUDA 版本。
- 从 Hugging Face 下载 4B 和 32B，固定 revision，逐文件校验 sha256，不需要 token。4B 总是会下载，即使写了 `--model 32b`。
- 检查数据包；检查主机：驱动、内存、`/dev/shm`、多卡通信。

最后一行是 `SETUP OK` 才算通过。如果是 `SETUP INCOMPLETE`，按上面的报错修完再跑一遍，已完成的部分不会重做。

用的是机器上已有的 conda。没被找到时先 `export CONDA_BASE=/path/to/conda`。完全没有 conda 时脚本会把 Miniforge 装到
`./.conda`。耗时主要看网速，32B 的下载最久。

### 3. 先看计划，不占 GPU

```bash
bash pipelines/sweep.sh --dry-run
```

逐个组合打印：用哪几张卡、每步多少条、多少步、每个框架的并行方式。H200 上 32B 应该是：ms-swift / TRL / Axolotl 显示
`replicate`，NeMo 显示 `fsdp2`，Unsloth 显示 `replicate, bf16`。出现 `SKIP` 或 `4-bit` 说明脚本认为单卡放不下模型，先停下来查原因。

### 4. 冒烟：每个组合 5 步训练 + 50 张图评测

```bash
nohup bash pipelines/sweep.sh --steps 5 --eval n50 > runs/sweep_smoke.out 2>&1 &
tail -f runs/sweep_smoke.out
```

每个框架第一次遇到一个模型时，会先在单卡上做一次正确性检查（核对输入、训练的 token、可训练参数、第 1 步 loss），之后用缓存。
结束时打印汇总表，全部是 `OK` 再往下走。这一轮顺便量出了每步耗时，可以用来估算正式跑要多久：

```bash
bash pipelines/run.sh status                 # 表里有 step s、tokens/s、峰值显存
```

只想先试一个组合：

```bash
bash pipelines/run.sh train --fw swift --model 32b --gpus 8 --steps 5 --eval n50
```

### 5. 正式跑

```bash
nohup bash pipelines/sweep.sh > runs/sweep.out 2>&1 &
tail -f runs/sweep.out
```

默认是 4B、32B × 2 / 4 / 6 / 8 张卡 × 5 个框架，每个都是完整 1 个 epoch 加 1,100 张测试图的评测，一共 40 次。

- **缩小范围**：`MODELS="32b" GPU_COUNTS="4 8" bash pipelines/sweep.sh --fw swift,trl`。其余参数原样传给 `run.sh train`。
- **失败不中断**：某个组合失败或卡数不够，后面的组合照常跑，结束时汇总表里标出来。
- **6 张卡**：每步是 30 条而不是 32 条，和 2 / 4 / 8 张卡的结果不完全可比。
- **停止**：输出的第一行是一条 `kill -TERM -- -<进程组号>` 命令，在另一个终端执行它，正在训练的进程会一起停掉。
  前台运行时直接 Ctrl-C。

### 6. 看结果

```bash
bash pipelines/run.sh status
```

| 想看什么 | 在哪里 |
|---|---|
| 所有训练的速度、显存、F1@0.5 及置信区间 | `runs/leaderboard.md` |
| 训好的 LoRA（标准 HF PEFT 格式） | `runs/<框架>/<模型>-<策略>-<N>g-run-<时间>/adapter/` |
| 某次训练的日志、评测结果 | 同一目录下的 `train.log`、`eval/full/metrics.json` |
| sweep 里每个组合的完整输出 | `runs/sweep_<时间>/<模型>_<N>g.log` |

### 7. 出问题时看哪里

| 现象 | 日志 |
|---|---|
| `setup` 里某个 env 建不起来 | `.cache/setup_<框架>.log` |
| 模型下载失败 | `.cache/download_Qwen3-VL-*.log` |
| 正确性检查 FAIL | `runs/audit_<框架>_<模型>.log` |
| 训练 FAIL | 该次训练目录下的 `train.log` 末尾 |
| 评测 FAIL | 该次训练目录下的 `eval_<子集>.log` |
| 每一步的结果流水 | `runs/run_status.tsv` |

## 验证范围

这套代码在一台 4 × RTX 6000 Ada（48GB）的机器上验证过（2026-10-11）：

- 4B：1、2、3、4 张卡 × 5 个框架的短训练全部通过；4 张卡 20 步训练 + 100 张图评测，5 个框架全部通过。
- 32B：ms-swift 用 4 张卡、分片模式的短训练通过。
- 4B 从 Hugging Face 自动下载并校验通过。

那台机器只有 4 张 48GB 的卡，所以下面这些没有实测过，在 H200 上第一次跑时要留意（第 4 步的冒烟就是为此准备的）：

- 6 张卡、8 张卡。
- 32B 每卡放一份完整模型的模式（DDP），H200 上默认走这个；NeMo 的 32B。
- 默认配置的完整 epoch + 全部测试图评测，只跑过短步数的版本。
- 没有 conda 的机器上自动装 Miniforge；`/dev/shm` 很小时自动关掉 DataLoader worker。
- 只有代码的新机器上自动生成数据包。

已知情况：

- 4B 训练 20 步后，100 张图上的 F1 是 0.64–0.66，低于不微调的 base 模型（0.736）；更早一次完整 epoch 的结果约 0.71。
  100 张图的误差约 ±0.1，所以微调是否带来提升还没有结论，要看完整 epoch + 全部测试图的结果。
- TRL 训练 32B 时，在 48GB 卡的分片模式下显存不够，计划里会标 `SKIP`。H200 上不走这条路径。
