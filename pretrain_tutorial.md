# SpongeBob-Pro 新环境配置与预训练验收指南

这份文档面向学生和 Codex，目标是在**现成 Python 环境**里，以尽量少的步骤完成预训练环境验收。

建议优先使用教学脚本：

```bash
bash scripts/run_pretrain_demo.sh
```

这个脚本会帮你：

1. 复用当前环境变量；
2. 组织常用训练参数；
3. 按默认配置直接拉起单卡或双卡预训练；
4. 方便通过环境变量覆盖关键超参数。

如果你想手动理解每一步，再继续读下面内容。

## 1. 核心原则

- 预训练主入口只看 `train/pretrain.py`；
- 数据下载是环境验收的一部分，不应跳过；
- `scripts/run_pretrain_demo.sh` 是教学脚本，不是排障主入口；
- 启动验证只需要看到训练**开始打印 step**，不要求完整跑完一个 epoch；
- 验收时优先使用**保守 batch size**，先保证成功启动，再考虑调大吞吐。

## 2. 先确认当前 Python 环境

```bash
python --version
which python
python -m pip --version
```

默认直接复用当前镜像环境，不额外新建虚拟环境。后续所有命令都要使用同一个 `python` 和同一个 `pip`。

## 3. 建议先固定两个环境变量

```bash
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

原因：

- 某些环境会把 `OMP_NUM_THREADS` 设成非法值，导致 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`；
- `expandable_segments:True` 可以减少 CUDA 显存碎片，对启动阶段更稳。

## 4. 检查并安装依赖

先检查缺包：

```bash
python - <<'PY'
import importlib.util

packages = [
    "torch",
    "numpy",
    "tqdm",
    "transformers",
    "tokenizers",
    "modelscope",
    "swanlab",
]

missing = [name for name in packages if importlib.util.find_spec(name) is None]
print("all_required_packages_installed" if not missing else f"missing: {', '.join(missing)}")
PY
```

如果缺包，优先用清华源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果外网连接失败，可以重试：

```bash
source /etc/network_turbo
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
unset http_proxy && unset https_proxy
```

安装后复查：

```bash
python - <<'PY'
import numpy, torch, tqdm, transformers, tokenizers, modelscope, swanlab
print("dependency_check=ok")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
```

## 5. 只下载训练必需的数据文件

远端数据集：

```text
Harris/SpongeBobPRO
```

不要整仓下载；验收环境时只需要预训练用的两个文件：

```bash
mkdir -p data/pretrain_data
modelscope download \
  --dataset Harris/SpongeBobPRO \
  --local_dir data/pretrain_data \
  SpongeBobPRO_pretrain_512_final.bin \
  SpongeBobPRO_pretrain_512_final.meta
```

这样更快，也避免把无关的权重和大 JSONL 一起拉下来。

## 6. 规范化训练文件名

训练脚本期望：

```bash
data/pretrain_data/spongebob_pretrain_512.bin
data/pretrain_data/spongebob_pretrain_512.meta
```

推荐使用**可重复执行**的规范化方式，而不是要求目录里“恰好只有一个 `.bin` 和一个 `.meta`”：

```bash
cp -f data/pretrain_data/SpongeBobPRO_pretrain_512_final.bin \
  data/pretrain_data/spongebob_pretrain_512.bin
cp -f data/pretrain_data/SpongeBobPRO_pretrain_512_final.meta \
  data/pretrain_data/spongebob_pretrain_512.meta

ls -lh data/pretrain_data/spongebob_pretrain_512.bin
ls -lh data/pretrain_data/spongebob_pretrain_512.meta
```

这样脚本多次执行也不会因为目录里已经有规范化后的文件而误报。
如果你想保持目录整洁，也可以在确认拷贝成功后手动删除原始下载文件；保留它们也不会影响后续训练。

## 7. 单卡训练验证

推荐先用保守配置做 smoke test：

```bash
python train/pretrain.py \
  --device cuda:0 \
  --data_path data/pretrain_data/spongebob_pretrain_512.bin \
  --save_dir pretrain_out/verify_single \
  --epochs 1 \
  --global_batch_size 8 \
  --head_size 64 \
  --learning_rate 1e-3 \
  --log_interval 1 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

为什么这里用 `global_batch_size=8`：实际验收时，`128` 在 87M 模型上可能直接触发 CUDA OOM；`8` 更适合作为“环境是否通”的默认值。

### 单卡成功标志

- 日志出现 `Dataset loaded:`
- 日志出现 `Starting training:`
- 日志开始打印 `Epoch:[...]`

只要这些出现，就说明单卡环境已经配置成功；此时可以手动停止。

## 8. 双卡训练验证

前提：

```bash
python - <<'PY'
import torch
count = torch.cuda.device_count()
print("cuda_device_count =", count)
if count < 2:
    raise SystemExit("Need at least 2 visible GPUs for the dual-GPU test.")
PY
```

双卡也建议先用保守全局 batch size：

```bash
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 2 \
  --master_port 29500 \
  train/pretrain.py \
  --data_path data/pretrain_data/spongebob_pretrain_512.bin \
  --save_dir pretrain_out/verify_ddp2 \
  --epochs 1 \
  --global_batch_size 8 \
  --head_size 64 \
  --learning_rate 1e-3 \
  --log_interval 1 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

这里双卡时每张卡实际 batch size 是 `4`。  
环境验收阶段，目标是“能稳定启动并打印 step”，不是追求吞吐最大化。

### 双卡成功标志

- 两个 rank 都正常启动
- 没有分布式初始化错误
- 日志开始打印训练 step

确认后即可手动停止。

## 9. 教学脚本的推荐用法

默认直接用脚本启动：

```bash
bash scripts/run_pretrain_demo.sh
```

如果你只想跑单卡：

```bash
NPROC_PER_NODE=1 bash scripts/run_pretrain_demo.sh
```

如果你想覆盖关键模型结构参数：

```bash
HIDDEN_SIZE=768 NUM_ATTENTION_HEADS=12 HEAD_SIZE=64 bash scripts/run_pretrain_demo.sh
```

## 10. 常见问题

- `libgomp: Invalid value for environment variable OMP_NUM_THREADS`
  - 先执行 `export OMP_NUM_THREADS=1` 再重试。
- `ModuleNotFoundError`
  - 说明依赖没有安装到当前 `python` 对应的环境里。
- `modelscope: command not found`
  - 说明 `modelscope` 没有安装成功，重新执行依赖安装。
- `data file not found`
  - 先确认是否完成下载，以及是否已经复制成 `spongebob_pretrain_512.bin/.meta`。
- CUDA OOM
  - 先减小 `global_batch_size`，环境验收建议从 `8` 开始。
- 双卡不满足条件
  - 单卡仍可验收基础环境，但不能证明 DDP 配置正确。

## 11. 验收完成的定义

下面四件事都成立，才算真正串起来：

1. 依赖导入检查通过；
2. 正式预训练数据已下载并放到正确路径；
3. 单卡训练已进入训练循环并打印 step；
4. 若机器可见 GPU 数量不少于 2，双卡 DDP 训练也已进入训练循环并打印 step。

如果这四点都满足，就可以认为环境、依赖、数据、训练入口已经配置成功。
