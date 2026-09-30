# My Jev

[中文](#中文) · [English](#english) · [Validation results / 验证结果](#validation-results--验证结果)

## 中文

基于 [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) 的文本决策模型：**Qwen3.5-2B + LoRA + 单标量评分头**，支持 Choice、Score 和 Noul。仓库包含源码、配置、测试、示例与文档；模型权重、数据集、运行结果和环境缓存不纳入 Git。

### 模型与数据

- 基础模型：[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，仅使用文本 backbone；LoRA rank=8、alpha=16，评分头采用 Yes−No 初始化。
- 数据集：[ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev)，配置 `release-v2-redistributable`，保留原始五个 split 和软标签。
- train / validation / calibration / test / ood 分别为 **79,116 / 3,723 / 4,672 / 10,356 / 15,701** 条决策。
- 模型输入仅由 `state/question/kind/options` 构造。候选独立评分，每条决策使用 soft cross-entropy + 0.1 × Brier；只更新 LoRA 和评分头。

Choice 返回概率最大的候选；Score 返回从 0 开始的期望等级；Noul 返回 `P(true)`。模型、数据 revision 固定在 [配置文件](configs/qwen35_2b_gpu32.yaml) 中。本项目每轮均匀打乱完整 train；上游 pilot 使用按 source/kind 均衡的循环采样。

### 安装与下载

在 Linux NVIDIA GPU 机器上执行。已验证环境为 Python 3.12、PyTorch 2.8.0/cu128、Transformers 5.10.2；依赖通过 `uv.lock` 固定，默认配置在 32 GiB GPU 上验证过。

```bash
cd my_jev
bash scripts/setup_server.sh
source scripts/env.sh
uv run --no-sync python scripts/prepare_data.py
uv run --no-sync python scripts/download_model.py
uv run --no-sync python scripts/preflight.py
```

`setup_server.sh` 会按需安装 uv 并创建环境。数据保存到 `data/`，基础模型保存在 Hugging Face 缓存；`env.sh` 默认将环境下载缓存放到项目同级的 `cache/`。每次开启终端后先加载 `scripts/env.sh`。

AutoDL 下载加速可选：将两条下载命令分别加上 `bash scripts/with_network_turbo.sh` 前缀；加速仅作用于下载子进程，退出时清除代理变量。

### 训练、续训与推理

```bash
# 默认：完整训练集 1 轮，共 2,473 次更新。
uv run --no-sync python train.py --config configs/qwen35_2b_gpu32.yaml

# 将 RUN 替换为训练结束后打印的实际运行目录。
RUN="runs/qwen35_2b_gpu32_YYYYMMDD_HHMMSS"
uv run --no-sync python infer.py --run "$RUN" --request examples/request.json
uv run --no-sync python eval.py --run "$RUN" --splits test ood

# 从 latest 续训；可将总训练轮数延长到 2。
uv run --no-sync python train.py --config "$RUN/config.json" --resume "$RUN" --set train.epochs=2
```

默认决策 `batch_size=32`、`accumulation=1`、候选序列并行上限 `candidate_batch_size=16`。有效 batch 为 `batch_size × accumulation`；LoRA/head 学习率分别为 `5e-5 / 1e-4`。可修改 YAML 或用 `--set train.batch_size=16` 等覆盖参数；除增加总轮数外，改变实验条件需新开运行。

训练、验证与校准均显示进度条。训练按 validation 未校准 NLL 选择 `best.pt`，结束后只用 calibration 拟合温度；`infer.py` 默认使用校准后的 best。test/OOD 需显式执行评估命令。请求格式见 [示例](examples/request.json)。

每个运行保留最近 3 份周期 checkpoint 和大型评估结果，另保留 `latest.pt` 与 `best.pt`。latest 包含优化器、随机状态和训练游标，用于断点续训；best 用于推理。基础模型不重复写入 checkpoint。

## English

A text decision model based on [Open-Jev](https://github.com/Zefan-Cai/Open-Jev): **Qwen3.5-2B + LoRA + a scalar scoring head**, supporting Choice, Score and Noul. This repository contains source code, configurations, tests, examples and documentation. Model weights, datasets, run artifacts and environment caches are excluded from Git.

### Model and dataset

- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), using its text backbone with LoRA rank 8, alpha 16 and a Yes−No initialized scoring head.
- Dataset: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), configuration `release-v2-redistributable`; original splits and soft targets are preserved.
- train / validation / calibration / test / ood contain **79,116 / 3,723 / 4,672 / 10,356 / 15,701** decisions.
- Prompts use only `state/question/kind/options`. Candidates are scored independently. Each decision contributes soft cross-entropy + 0.1 × Brier loss; only LoRA and the scoring head are updated.

Choice selects the highest-probability candidate, Score returns the expected zero-based level, and Noul returns `P(true)`. Model and dataset revisions are pinned in the [configuration](configs/qwen35_2b_gpu32.yaml). This implementation shuffles the full training split each epoch; the upstream pilot uses source/kind-balanced cyclic sampling.

### Setup and downloads

Run on a Linux machine with an NVIDIA GPU. The verified environment uses Python 3.12, PyTorch 2.8.0/cu128 and Transformers 5.10.2, pinned through `uv.lock`. The default configuration was verified on a 32 GiB GPU.

```bash
cd my_jev
bash scripts/setup_server.sh
source scripts/env.sh
uv run --no-sync python scripts/prepare_data.py
uv run --no-sync python scripts/download_model.py
uv run --no-sync python scripts/preflight.py
```

The setup script installs uv if needed and creates the environment. Data is stored in `data/`; base weights use the Hugging Face cache. By default, `env.sh` places download caches in a sibling `cache/` directory. Source `scripts/env.sh` in each new terminal session.

For optional AutoDL download acceleration, prefix each download command with `bash scripts/with_network_turbo.sh`. Proxy settings are limited to the download subprocess and cleared when it exits.

### Training, resuming and inference

```bash
# One full training epoch: 2,473 optimizer updates.
uv run --no-sync python train.py --config configs/qwen35_2b_gpu32.yaml

# Replace RUN with the actual directory printed by training.
RUN="runs/qwen35_2b_gpu32_YYYYMMDD_HHMMSS"
uv run --no-sync python infer.py --run "$RUN" --request examples/request.json
uv run --no-sync python eval.py --run "$RUN" --splits test ood

# Resume from latest and extend the total budget to two epochs.
uv run --no-sync python train.py --config "$RUN/config.json" --resume "$RUN" --set train.epochs=2
```

Defaults: decision `batch_size=32`, `accumulation=1`, and at most 16 candidate sequences per backbone call. Effective batch size is `batch_size × accumulation`; LoRA/head learning rates are `5e-5 / 1e-4`. Edit YAML or use overrides such as `--set train.batch_size=16`. Changing experiment conditions requires a new run, except when increasing the total epoch budget.

Training, validation and calibration have progress bars. The best checkpoint minimizes uncalibrated validation NLL; a temperature is then fitted on calibration only. Inference defaults to the calibrated best checkpoint. Test/OOD evaluation is explicit. See the [request example](examples/request.json).

Each run retains the three latest periodic checkpoints and large evaluation artifacts, plus `latest.pt` and `best.pt`. Latest contains optimizer, RNG and cursor state for resuming; best is used for inference. Base weights are not duplicated in checkpoints.

## Validation results / 验证结果

完整训练一轮的最后一次验证：**step 2473、seed 42、3,723 条 validation 决策、T=1（未校准）**。结果来自本项目实际训练，不是上游报告或 smoke 测试。

Final validation after one full epoch: **step 2473, seed 42, 3,723 validation decisions, T=1 (uncalibrated)**. These are measured results from this implementation, not upstream claims or smoke-test scores.

| Task / 任务 | Decisions (hard targets) / 样本数（硬标签） | Hard accuracy ↑ / 硬标签准确率 | Expected accuracy ↑ / 期望准确率 | NLL ↓ | Task metric / 任务指标 |
|---|---:|---:|---:|---:|---:|
| Choice | 957 (740) | 82.03% | 72.14% | 0.5845 | — |
| Score | 481 (470) | 95.53% | 94.32% | 0.1124 | Ordinal MAE ↓: 0.0588 |
| Noul | 2,285 (2,285) | 96.54% | 96.54% | 0.1140 | F1 ↑: 93.81% |

硬标签准确率只统计 one-hot target；期望准确率在全部记录上平均所选类别的 target 概率。Score 准确率比较概率最大等级，MAE 比较预测与 target 的期望等级；Noul F1 使用 `P(true) ≥ 0.5`。

Hard accuracy uses one-hot targets only; expected accuracy averages target probability at the selected class over all decisions. Score accuracy compares the most probable level, while MAE compares predicted and target expected levels. Noul F1 uses `P(true) ≥ 0.5`.

最终整体硬标签准确率为 **93.33%**，NLL 为 **0.2348**。按整体 NLL 选出的 best 是 **step 1875**：整体准确率 **93.79%**、NLL **0.2158**；其 Choice / Score / Noul 硬标签准确率分别为 **81.76% / 96.60% / 97.11%**。推理默认使用该 best，校准温度为 **1.11267**。本次仅一个 seed，尚未完成全量 test/OOD 评估。

Final overall hard accuracy is **93.33%**, with NLL **0.2348**. The best checkpoint by overall NLL is **step 1875**: overall accuracy **93.79%**, NLL **0.2158**, and Choice / Score / Noul hard accuracy **81.76% / 96.60% / 97.11%**. Default inference uses this best checkpoint with temperature **1.11267**. This is a single-seed run; full test/OOD evaluation has not been performed.

## Repository layout / 仓库结构

```text
configs/        Training configurations / 训练配置
src/my_jev/     Model, losses, training and recovery / 模型、损失、训练与恢复
scripts/        Setup, downloads, checks and profiling / 环境、下载、检查与资源测量
tests/          Correctness and recovery tests / 正确性与恢复测试
examples/       Inference request example / 推理请求示例
train.py        Training entry point / 训练入口
infer.py        Inference entry point / 推理入口
eval.py         Test/OOD evaluation / 独立评估
uv.lock         Locked dependencies / 依赖锁
```

在计算服务器上运行测试 / Run checks on the compute server:

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest
```

Third-party sources and licenses / 第三方来源与许可证：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) · [Open-Jev MIT license](src/my_jev/_upstream/LICENSE).
