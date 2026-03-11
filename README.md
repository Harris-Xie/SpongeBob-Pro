# SpongeBob-Pro

SpongeBob-Pro 是一个面向中文场景的小型语言模型实验仓库，覆盖从 tokenizer、数据预处理、预训练，到 SFT、GRPO 与基准评测的完整训练链路。当前实现以 `model/` 中的 decoder-only Transformer 为核心，支持 RoPE、GQA 和 PyTorch 2.x 的 Flash Attention 路径，适合教学演示、环境验收和小模型训练实验。

## 项目概览

- `model/`: 模型配置与网络实现，核心文件是 `config.py` 和 `model_spongebob_pro.py`
- `dataset/`: 预训练二进制数据集、SFT/GRPO 数据集，以及预处理脚本
- `train/`: 预训练、SFT、GRPO、Tokenizer 训练入口
- `benchmark/`: C3、XCOPA 与 `mini_bench` 评测脚本
- `scripts/`: 推荐使用的启动脚本，当前主要是预训练 smoke test 入口

当前仓库主要支持三类训练任务：

1. 预训练：基于 `.bin/.meta` 的高吞吐数据读取和 DDP 训练。
2. SFT：基于对话 `jsonl` 数据做监督微调，只计算 assistant 部分 loss。
3. GRPO：对 `<think>...</think>` 格式输出进行约束，并调用外部 Judge 计算奖励。

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果需要下载预训练验收数据，仓库默认使用 ModelScope 数据集：

```bash
mkdir -p data/pretrain_data
modelscope download \
  --dataset Harris/SpongeBobPRO \
  --local_dir data/pretrain_data \
  SpongeBobPRO_pretrain_512_final.bin \
  SpongeBobPRO_pretrain_512_final.meta

cp -f data/pretrain_data/SpongeBobPRO_pretrain_512_final.bin \
  data/pretrain_data/spongebob_pretrain_512.bin
cp -f data/pretrain_data/SpongeBobPRO_pretrain_512_final.meta \
  data/pretrain_data/spongebob_pretrain_512.meta
```

最短预训练验证方式：

```bash
NPROC_PER_NODE=1 GLOBAL_BATCH_SIZE=8 USE_SWANLAB=0 EVAL_BENCH=0 \
bash scripts/run_pretrain_demo.sh
```

如果你希望手动运行预训练入口：

```bash
python train/pretrain.py \
  --device cuda:0 \
  --data_path data/pretrain_data/spongebob_pretrain_512.bin \
  --save_dir pretrain_out/verify_single \
  --epochs 1 \
  --global_batch_size 8 \
  --head_size 64 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

更完整的环境验收说明见 [`pretrain_tutorial.md`](./pretrain_tutorial.md)。

## 训练与评测入口

- 预训练: `python train/pretrain.py --help`
- SFT: `python train/train_sft.py --help`
- GRPO: `python train/train_grpo.py --help`
- Tokenizer: `python train/train_tokenizer.py`
- 预处理: `python dataset/preprocess_data.py --help`
- mini bench: `python benchmark/mini_bench/run_test.py --max_prompts 5`

常用环境变量：

- `SWANLAB_API_KEY`: 预训练/SFT 日志上报使用
- `DEEPSEEK_API_KEY`: SFT 的 `mini_bench` Judge 或 GRPO 奖励计算使用

## 当前模型与训练特性

- 基础架构是 decoder-only Transformer
- 注意力实现支持 GQA
- 位置编码使用 RoPE
- 在可用环境下走 `scaled_dot_product_attention`
- 预训练支持单卡和单机多卡 DDP
- 预训练数据使用 memmap 读取 `.bin/.meta`
- SFT 与 GRPO 已提供独立数据管线和训练脚本

## `main` 到 `new_arch_dev` 的改动摘要

当前 `new_arch_dev` 分支相对 `main` 主要做了以下更新，README 里只保留最关键的信息：

- 模型结构从“默认头维度推导”改成“显式可配置头维度”。
  现在 `SpongeBobConfig` 和预训练入口新增了 `head_size`，注意力投影和 RoPE 维度都改为基于 `head_size` 计算，便于实验不同的注意力头配置。
- 预训练入口被改成更适合教学和验收的形态。
  现在统一使用 `global_batch_size`，补充了更完整的架构参数暴露、参数合法性校验、step 耗时日志、可配置 tokenizer/benchmark 路径，以及从环境变量读取 `SWANLAB_API_KEY`。
- 新增了一个可直接运行的预训练演示脚本。
  `scripts/run_pretrain_demo.sh` 支持单卡或多卡启动，并允许通过环境变量覆盖批大小、模型宽度、头数、`head_size` 等关键参数。
- 文档层面新增并重写了预训练验收说明。
  `pretrain_tutorial.md` 现在明确了依赖检查、ModelScope 数据下载、标准文件命名、单卡和双卡 smoke test 的推荐流程。
- 预训练 benchmark 路径和导入方式更稳定。
  当前分支修复了 benchmark 导入路径，避免依赖旧目录结构。

## 建议的最小验证

在提交代码或开始长任务前，至少执行一次：

```bash
python -m compileall model train dataset benchmark
```

如果你修改了训练或推理路径，建议再补一个小规模验证：

```bash
NPROC_PER_NODE=1 GLOBAL_BATCH_SIZE=8 USE_SWANLAB=0 EVAL_BENCH=0 \
bash scripts/run_pretrain_demo.sh
```

## 说明

- 仓库中部分脚本仍保留了本地实验路径或外部服务依赖，正式运行前需要按你的环境调整。
- 不要提交数据、权重、API Key 或机器相关的绝对路径。
