# Qwen3-VL LoRA SFT：5 个框架的训练 pipeline

同一份数据包（`data/package/vlm_sft_*`）、同一组超参（`common/hparams.sh`）、同样的 loss 掩码和图像分辨率，
分别用 **ms-swift / TRL+PEFT / NeMo AutoModel / Axolotl / Unsloth** 训练 LoRA；产出标准 HF PEFT adapter，
用同一个评测脚本比较效果、速度、显存。

## 目录

| 目录 | 内容 |
|---|---|
| `common/` | `env.sh`（路径、缓存、conda env）、`hparams.sh`（唯一超参来源）、`strategy.sh`（shard/replicate 自动选择）、`run_lib.sh`（run 目录、1 Hz 显存/内存监控、日志时间戳）、`hooks/sitecustomize.py`（步时/loss、参数清点、ViT 梯度探针、标签导出）、`audit.sh` + `reference_encode.py` + `compare_audit.py`（正确性检查）、`preflight.sh`、`zero3.json`（swift/TRL/Axolotl 共用的 ZeRO-3 配置） |
| `<fw>/` | `setup_env.sh` + `requirements.txt`（conda env `vlm<fw>_NV_py312`），`prep_data.py`（数据格式转换），`train.sh`（启动） |
| `eval/` | `predict.py`（严格加载 adapter、greedy 推理）、`score.py`（P/R/F1 + 置信区间）、`eval.sh`、`leaderboard.py` |
| `run.sh` | **统一入口**：`setup` / `train` / `status` |
| `sweep.sh` | 批量循环：模型 × 卡数 × 框架，逐个训练 + 评测 |
| `tools/` | `pack.sh`（打包代码和数据）、`sync.sh`（rsync 到远端） |

## 各框架怎么吃数据

| 框架 | 数据 | 转换脚本 |
|---|---|---|
| ms-swift | 直接用数据包 `train.jsonl`（`ROOT_IMAGE_DIR=$PKG`） | 只有 audit32 子集需要 `swift/prep_data.py` |
| TRL + PEFT | prompt/completion + `images` 列（绝对路径） | `trl/prep_data.py` |
| NeMo AutoModel | `{"conversation": [...]}`，由 `nemo/person_rock_data.make_dataset` 读成普通 list | `nemo/prep_data.py` |
| Axolotl | messages，图片写在消息里 `{"type":"image","path":绝对路径}` | `axolotl/prep_data.py` |
| Unsloth | messages + 顶层 `images` 列，消息里只放 `{"type":"image"}` 占位 | `unsloth/prep_data.py` |

转换结果写到 `data/formats/<pkg>/<fw>/`，`train.sh` 会自动调用（数据包路径或内容变了才重新生成）。

## 使用：三条命令

```bash
# 1) 每台机器一次：检查主机、建 5 个 conda env、准备模型和数据（缺什么补什么，可重复运行）
bash pipelines/run.sh setup                      # 默认准备 4B 和 32B；--model 4b 只准备 4B
                                                 # 4B 总是会随 env 一起从 Hugging Face 自动下载（约 9GB）

# 2) 端到端：正确性检查 → 训练 → 评测 → leaderboard
bash pipelines/run.sh train --fw swift --model 4b  --gpus 4
bash pipelines/run.sh train --fw all   --model 32b --gpus 8      # 5 个框架依次跑

# 3) 看结果
bash pipelines/run.sh status
```

`train` 会自己补齐前置条件（env 没建就建、模型没有就下载并校验 sha256、数据包没有就从 COCO 构建），
所以在一台只有代码的新机器上直接跑第 2 条也可以。

**模型下载**：`setup`（包括 `sweep.sh` 开头那次）建完 env 后总会确保 `models/Qwen3-VL-4B-Instruct` 在本地，没有就用
固定 revision 从 Hugging Face 下载并逐文件校验 sha256，`--model 32b` 时也一样；没跑过 `setup` 直接 `train`、env 是当场
建的，也会顺带下载 4B。已经拷好的模型目录只做校验，不重复下载。需要镜像或代理时照常设 `HF_ENDPOINT` / `HTTPS_PROXY`。

| 选项 | 含义 | 默认 |
|---|---|---|
| `--fw` | `swift` `trl` `nemo` `axolotl` `unsloth`，逗号分隔的列表，或 `all` | `train` 必填；`setup` 默认 `all` |
| `--model` | `4b` / `32b`（对应 `models/Qwen3-VL-<N>B-Instruct`），或模型目录 | `32b` |
| `--gpus` | 用几张卡，1–8（取可见卡里的前 N 张） | 全部 |
| `--gpu-ids` | 指定用哪几张卡，例如 `4,5,6,7`（可以和 `--gpus` 一起用：取其中前 N 张） | — |
| `--steps` | 训练步数（冒烟测试用） | 完整 1 个 epoch |
| `--lr` | 学习率 | 5e-5 |
| `--eval` | 训练后评测的测试子集：`full`（1,100 张）/ `n100` / `n50` / `none` | `full` |
| `--save-steps` | 每 N 步额外保存一次 checkpoint（长时间训练建议设 50） | 只保存最终 adapter |
| `--timeout` | 单次训练的墙钟上限，例如 `12h`；`0` 不限 | 不限 |
| `--dry-run` | 只打印计划（用哪几张卡、batch、步数、每个框架的并行方式）然后退出，不下载、不训练 | — |
| `--strategy` | 强制 swift / trl / axolotl 的并行策略 `shard` / `replicate`（NeMo 永远用 FSDP2 分片，Unsloth 永远 DDP） | 按显存自动选 |
| `--no-audit` / `--no-baseline` | 跳过正确性检查 / 跳过 base 模型的 zero-shot 基线 | 都做 |

要点：

- **退出码**：0 = 要求的框架全部成功；1 = 至少一个失败；2 = 参数或前置条件错误。计划阶段判定为不支持的组合
  记为 SKIP，不算失败。
- **开工前先打印计划**：模型、用哪几张卡、实际 batch、步数，以及每个框架将用的并行方式；放不下的组合在这里就被拦下。
- **批大小**：每卡每次 1 条，梯度累积取 round(32 / 卡数)。1、2、4、8 张卡时每步正好 32 条；3、5、6、7 张卡时是
  33、30、30、35 条（会打印提示，leaderboard 里有 batch 列；这几种卡数的结果和其他卡数不完全可比）。
- **正确性检查**在单卡上做，结果按“框架 + 模型 + 数据内容 + 代码版本 + env 版本”缓存，都没变就不重复跑。所选模型
  单卡放不下时（例如 48GB 卡上的 32B），自动改用同系列的 4B 来检查（本地没有就先下载）。
- **训练后检查**：每次真实训练结束都会核对保存下来的 adapter（LoRA 模块数、参数量、没有落在 ViT / merger /
  lm_head 上）以及冻结的 ViT 没有被反传；不过就算训练失败。
- **成功的定义**：训练返回 0、`adapter/` 里有 PEFT adapter、训练后检查通过。
- **中断**：前台运行时按 Ctrl-C；后台运行时向整个进程组发信号（`kill -TERM -- -<进程组号>`），训练进程会一起停掉，
  不留占着 GPU 的孤儿进程。只 `kill` `run.sh` 这一个 pid 不保证立即停（没有验证过）。
- 底层脚本仍然可以单独用：`MODEL=4b NGPU=2 MAX_STEPS=20 bash pipelines/<fw>/train.sh`、
  `bash pipelines/common/audit.sh <fw> 4b`、`bash pipelines/eval/eval.sh <run_dir> full`。它们认的环境变量：
  `MODEL` `NGPU` `CUDA_VISIBLE_DEVICES` `MAX_STEPS` `LR` `SAVE_STEPS` `RUN_TIMEOUT` `STRATEGY` `QUANT`（仅 Unsloth）
  `AUDIT_GPU`（检查用哪张卡）。
- `DATALOADER_WORKERS`（默认 4）对 `run.sh` 也生效；容器里 `/dev/shm` 小于 4GB 时自动设为 0（否则 worker 会
  因共享内存不足报 Bus error），不影响训练结果，只影响数据加载速度。

## 批量跑：`sweep.sh`

```bash
bash pipelines/sweep.sh                         # 4b、32b × 2/4/6/8 张卡 × 5 个框架：完整 1 个 epoch + full 评测
bash pipelines/sweep.sh --dry-run               # 只打印每个组合的计划
bash pipelines/sweep.sh --steps 5 --eval n50    # 同一个循环的冒烟版（先跑这个，顺便量出每步耗时）
MODELS="32b" GPU_COUNTS="4 8" bash pipelines/sweep.sh --fw swift,trl
```

- `MODELS`（默认 `4b 32b`）和 `GPU_COUNTS`（默认 `2 4 6 8`）决定循环范围；其余参数原样传给 `run.sh train`。
- 开始前先跑一次 `setup`；某个组合失败或在本机跑不了（比如卡数不够）不会中断后面的组合。
- 每个组合的输出在 `runs/sweep_<时间>/<model>_<N>g.log`，结束时打印汇总；所有 run 照常进 leaderboard。
- 6 张卡时每步是 30 条而不是 32 条（见上面“批大小”），和 2/4/8 张卡的结果不完全可比。
- 长时间跑建议放后台：`nohup bash pipelines/sweep.sh > runs/sweep.out 2>&1 &`。输出的第一行是停止整个 sweep 的命令
  （`kill -TERM -- -<进程组号>`），正在训练的进程会一起停掉。

## 在 H200 pod 上

从打包、`setup`、冒烟到正式跑的完整步骤，以及哪些情况还没有实测过，见仓库根目录的 [README.md](../README.md)。

- pod 上没有 conda 时，`run.sh` 会把 Miniforge 装到 `./.conda`；已有 conda 就用已有的（`CONDA_BASE` 可指定）。
- 模型也可以自己拷到 `models/Qwen3-VL-32B-Instruct/`，`setup` 只做校验。
- H200 单卡放得下 32B，所以 swift / TRL / Axolotl / Unsloth 走 DDP（每卡一份完整 BF16 模型），正确性检查直接用 32B 做。

## 正确性检查发现的问题和修正

`common/audit.sh <fw> <4b|32b>` 在单卡上（模型必须单卡放得下）用 audit32（固定 32 条，含负样本和最拥挤的图）、框架**自己的**数据管线跑 1 步，
和参考编码（HF processor + Qwen3-VL 模板）逐项比对：input ids、训练 token、图像 token 数、可训练参数清点、
冻结 ViT 是否被反传、第 1 步 loss（LoRA B=0，所以等于 base 模型在这 32 条上按 token 加权的 CE）。
默认配置下 5 个框架里有 4 个**不在做同一件事**，修正如下（都在 pipeline 自己的目录里，不改框架安装）：

| 框架 | 默认行为的问题 | 修正 |
|---|---|---|
| TRL 1.15 | transformers 给 Qwen3-VL 设了 `accepts_loss_kwargs=False`，Trainer 不传 `num_items_in_batch`，loss 变成**按样本平均**（第 1 步 0.6176 vs 按 token 0.4779） | `trainer.model_accepts_loss_kwargs = True`（`trl/train.py`） |
| ms-swift 4.5.3 | `enable_input_require_grads()` 也给 ViT 的 patch embedding 挂了 hook，之后的 `disable` 删不掉 → 每步对冻结 ViT 白做一次反传 | `swift/no_vit_backward.py`（`--external_plugins`），显存 12.05→11.36 GiB，loss 不变 |
| Axolotl 0.20.0 | ① `role_boundaries` 行是 `DictDefault`，`hasattr(spec,"model_dump")` 误判 → 崩；② 同 TRL，按样本平均；③ 默认向 Axolotl 上报遥测 | `axolotl/vlm_fixes.py` 插件（转 dict + `model_accepts_loss_kwargs=True`）；`env.sh` 设 `AXOLOTL_DO_NOT_TRACK=1` |
| Unsloth 2026.10 | collator 默认 `resize="min"` → Qwen3-VL 回落到 512 宽；transformers 5.17 的 `min_pixels` 是只读属性 | `resize="max"`，图片放顶层 `images` 列；只设 `processor.size` |
| NeMo 0.6.0 | LoRA 默认 bf16 + xavier（lr 5e-5 时更新被舍入）；无 lr_scheduler 时 lr 恒定；分布式超时 1 分钟；DDP 下 fp32 LoRA 直接 dtype 报错 | yaml 显式 `lora_dtype: float32`、`lora_A_init: kaiming`、cosine 调度、30 分钟超时；任何情况都用 FSDP2 全分片（它也不允许 shard 组大小为 1 的 HSDP） |
| 全部 | popos-6000b 上卡间直接拷贝（`device_map=auto`）数据损坏（NCCL 正常） | `common/p2p_safe.py`：`device_map=auto` 时跨卡 `.to()` 经 CPU 中转；`preflight.sh` 检查数据正确性 |

修正后 5 个框架的 4B 检查全部通过（第 1 步 loss 偏差 ≤0.26%）。剩下的、记录在案的差异：swift/TRL 训练 `<|im_end|>` 后的
`\n`，其他三个不训练（对准确率可忽略，loss 数值约差 1%）；NeMo 的 LoRA 初始化来自它自己的实现。

## 并行策略

| 框架 | replicate（H200 32B、6000b 4B） | shard（6000b 32B） |
|---|---|---|
| ms-swift | DDP | DeepSpeed ZeRO-3 |
| TRL | accelerate DDP（每个进程直接加载到自己的卡） | DeepSpeed ZeRO-3 |
| NeMo | FSDP2 全分片（没有不分片的模式） | FSDP2 全分片 |
| Axolotl | DDP | DeepSpeed ZeRO-3 |
| Unsloth | DDP，BF16 | 不支持分片；32B 在 48GB 卡上只能 4-bit QLoRA（仅参考） |

单卡放得下模型（BF16 体积 ×1.3 + 8GB）就 replicate，否则 shard；`STRATEGY` 可手动指定。

## 输出

每次运行一个目录 `runs/<fw>/<model>-<strategy>-<N>g-run-<时间>/`：

- `train.log`（每行带时间戳）、`cmd.txt`、渲染后的配置（`axolotl.yaml` / `nemo.yaml` / `accelerate.yaml`）
- `hparams.env`（这次实际用的超参）、`run_info.txt`、`pip_freeze.txt`
- `gpu.csv` / `host_mem.csv`（每秒一次，只记录这次用到的卡）
- `steps.jsonl`、`param_census.json`、`vision_grad.rank*.json`（钩子写的）、`post_train_check.log`
- `run_summary.json`（加载耗时、稳态步时间、samples/s、tokens/s、峰值显存和内存、首尾 loss）
- `adapter/`（HF PEFT）、`eval/<subset>/`（逐条预测、`metrics.json`）

`runs/leaderboard.md` 汇总所有 run；`runs/run_status.tsv` 是 `run.sh` 每一步的结果流水。

## 规矩

- 所有缓存都在 `$ROOT/.cache`，输出不得写进 Dropbox（每个脚本都检查）。
- 不开 W&B，不记录任何图片。
- 6000b 上 n100/n50 的准确率只用来看各框架是否“同一水平”，正式比较在 H200 上用 full 测试集看置信区间。
