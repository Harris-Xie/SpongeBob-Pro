# SpongeBob-Pro 新环境配置与预训练验收指南

这份文档既是给学生看的，也希望 Codex 在一个全新的环境里看到它后，能够按步骤完成下面这条链路：

1. 创建并配置 Python 环境；
2. 检查当前环境里缺少哪些依赖；
3. 按 `requirements.txt` 安装依赖；
4. 从 ModelScope 下载官方预训练数据；
5. 用 `train/pretrain.py` 完成单卡训练启动验证；
6. 在至少 2 张 GPU 可见时，完成双卡训练启动验证。

核心原则：

- 当前预训练主入口只看 `train/pretrain.py`；
- 数据需要单独下载，不放在训练启动脚本里自动处理；
- `scripts/run_pretrain_demo.sh` 只是教学脚本，方便理解参数组织方式，不是唯一入口；
- 下载训练数据本身，也是“环境是否完整配置好”的一部分；
- 对 Codex 来说，看到训练成功启动，就说明环境、依赖、数据路径和训练入口已经串起来了。

## 1. 创建 Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

建议后续所有命令都在该虚拟环境中执行。

## 2. 依赖文件说明

项目根目录提供了：

```bash
requirements.txt
```

它记录了当前预训练主流程需要的 Python 依赖：

- `torch`
- `numpy`
- `tqdm`
- `transformers`
- `tokenizers`
- `modelscope`
- `swanlab`

这里 `swanlab` 不是可选项，而是项目依赖的一部分，应当和其他依赖一起安装。

## 3. 先检查依赖，再安装依赖

在新环境里，推荐先检查当前环境是否已经具备所需依赖，而不是直接盲目安装。

### 3.1 检查当前环境是否缺包

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

if missing:
    print("missing:", ", ".join(missing))
else:
    print("all_required_packages_installed")
PY
```

如果输出 `all_required_packages_installed`，可以继续下一步。  
如果输出 `missing: ...`，就按下面方式安装。

### 3.2 按 `requirements.txt` 安装依赖

优先使用清华源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果清华源不可用，再退回默认 PyPI：

```bash
python -m pip install -r requirements.txt
```

### 3.3 安装后再次验证

```bash
python - <<'PY'
import numpy
import torch
import tqdm
import transformers
import tokenizers
import modelscope
import swanlab

print("dependency_check=ok")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
```

如果这里报错，就不要继续后面的步骤，先修复环境。

## 4. 理解项目里的关键入口

- `train/pretrain.py`：当前预训练主入口；
- `dataset/pretrain_dataset.py`：读取 `.bin + .meta` 预训练数据；
- `tokenizer_15k/`：训练和 benchmark 用 tokenizer；
- `data/pretrain_data/`：正式预训练数据目录；
- `pretrain_out/`：训练输出目录；
- `scripts/run_pretrain_demo.sh`：教学版启动脚本，本质上只是把参数整理后调用 `train/pretrain.py`。

这里最重要的一点是：

**现在不把 `train/pretrain_without_ddp.py` 当作主线。**

单卡和多卡都统一围绕 `train/pretrain.py` 来理解和验证。

## 5. 从 ModelScope 下载官方数据

远端数据集路径：

```text
Harris/SpongeBobPRO
```

先创建本地目录，再下载：

```bash
mkdir -p data/pretrain_data
modelscope download --dataset Harris/SpongeBobPRO --local_dir data/pretrain_data
```

下载完成后，建议先列出文件确认一下：

```bash
find data/pretrain_data -maxdepth 2 -type f | sort
```

## 6. 规范化训练所需的数据文件名

训练脚本期望的数据文件名是：

```bash
data/pretrain_data/spongebob_pretrain_512.bin
data/pretrain_data/spongebob_pretrain_512.meta
```

如果 ModelScope 下载后的文件名已经一致，可以跳过这一节。

如果名字不一致，可以用下面这段脚本做一次规范化拷贝。它会 fail-fast：如果目录里不是“恰好 1 个 `.bin` 和 1 个 `.meta`”，脚本会直接报错退出。

```bash
python - <<'PY'
from pathlib import Path
import shutil

root = Path("data/pretrain_data")
bin_files = sorted(root.rglob("*.bin"))
meta_files = sorted(root.rglob("*.meta"))

if len(bin_files) != 1:
    raise RuntimeError(f"Expected exactly 1 .bin file, found {len(bin_files)}: {bin_files}")
if len(meta_files) != 1:
    raise RuntimeError(f"Expected exactly 1 .meta file, found {len(meta_files)}: {meta_files}")

target_bin = root / "spongebob_pretrain_512.bin"
target_meta = root / "spongebob_pretrain_512.meta"

if bin_files[0] != target_bin:
    shutil.copy2(bin_files[0], target_bin)
if meta_files[0] != target_meta:
    shutil.copy2(meta_files[0], target_meta)

print("normalized:", target_bin)
print("normalized:", target_meta)
PY
```

然后验证文件存在：

```bash
ls -lh data/pretrain_data/spongebob_pretrain_512.bin
ls -lh data/pretrain_data/spongebob_pretrain_512.meta
```

## 7. 单卡训练验证

这一步的目标是：**用正式训练数据成功启动一次单卡训练。**

这里的“验证成功”不要求你完整跑完整个 epoch；  
只要训练命令成功启动，数据被正确加载，日志进入训练循环，就说明：

- Python 环境是好的；
- 依赖安装是好的；
- 数据下载是好的；
- 数据文件名和路径是好的；
- `train/pretrain.py` 这条主入口是好的。

推荐命令：

```bash
python train/pretrain.py \
  --device cuda:0 \
  --data_path data/pretrain_data/spongebob_pretrain_512.bin \
  --save_dir pretrain_out/verify_single \
  --epochs 1 \
  --global_batch_size 128 \
  --learning_rate 1e-3 \
  --log_interval 1 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

如果你只是在没有 GPU 的机器上做功能确认，也可以改成：

```bash
python train/pretrain.py \
  --device cpu \
  --data_path data/pretrain_data/spongebob_pretrain_512.bin \
  --save_dir pretrain_out/verify_cpu \
  --epochs 1 \
  --global_batch_size 8 \
  --learning_rate 1e-3 \
  --log_interval 1 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

### 单卡验证成功的标志

- 日志里出现 `Dataset loaded:`
- 日志里出现 `Starting training:`
- 日志里开始打印训练 step

一旦这些都出现，就说明单卡环境已经配置成功。  
如果只是验收环境，可以在确认训练已正常开始后手动停止。

## 8. 双卡训练验证

这一步的目标是：**验证 DDP 环境和 `torchrun` 配置是否正确。**

前提条件：

- 当前机器至少有 `2` 张可见 GPU；
- `python -c "import torch; print(torch.cuda.device_count())"` 输出不小于 `2`。

先检查 GPU 数量：

```bash
python - <<'PY'
import torch
count = torch.cuda.device_count()
print("cuda_device_count =", count)
if count < 2:
    raise SystemExit("Need at least 2 visible GPUs for the dual-GPU test.")
PY
```

然后执行双卡训练：

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
  --global_batch_size 256 \
  --learning_rate 1e-3 \
  --log_interval 1 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab 0 \
  --use_compile 0 \
  --eval_bench 0
```

这里 `global_batch_size=256`，双卡时每张卡实际 batch size 是 `128`。

### 双卡验证成功的标志

- 两个 rank 都能正常启动
- 日志里没有分布式初始化错误
- 日志进入训练循环并开始打印 step

一旦这些都出现，就说明双卡环境已经配置成功。  
如果当前目的只是验收环境，也可以在确认训练已正常开始后手动停止。

## 9. `scripts/run_pretrain_demo.sh` 的定位

可以把这个脚本当成教学辅助材料：

```bash
bash scripts/run_pretrain_demo.sh
NPROC_PER_NODE=2 GLOBAL_BATCH_SIZE=256 bash scripts/run_pretrain_demo.sh
```

它的价值在于：

- 帮学生理解“脚本如何组织参数”；
- 帮 Codex 快速复用一组默认参数；
- 但真正的训练入口和排障对象，仍然是 `train/pretrain.py`。

## 10. 建议 Codex 在新环境里的最小验收顺序

如果目标是在一个新环境里验证“这个项目已经被完整配置好”，建议按下面顺序执行：

1. 创建虚拟环境；
2. 检查当前环境缺少哪些依赖；
3. 按 `requirements.txt` 安装依赖；
4. 再次执行依赖导入检查；
5. 从 ModelScope 下载官方数据；
6. 规范化 `.bin/.meta` 文件名；
7. 跑单卡训练，直到确认训练成功启动；
8. 如果可见 GPU 数量不少于 2，再跑双卡训练，直到确认训练成功启动。

这里不要求先构造 smoke 数据。  
因为“能下载正式数据并用正式数据成功启动训练”，本身就是环境配置成功的重要组成部分。

## 11. 常见问题

- `ModuleNotFoundError`
  - 先回到第 3 节，检查依赖是否真的装进当前虚拟环境。
- `modelscope: command not found`
  - 说明 `modelscope` 没有安装成功，重新执行 `pip install -r requirements.txt`。
- `data file not found`
  - 先确认是否完成了 ModelScope 下载，以及文件名是否已经规范化。
- `Expected exactly 1 .bin file`
  - 说明下载目录里有多个候选文件；先手动检查 `find data/pretrain_data -type f` 的输出，再决定保留哪一份。
- `Need at least 2 visible GPUs`
  - 这说明当前机器不满足双卡验收条件；单卡训练仍可验证，但不能证明双卡环境可用。
- CUDA OOM
  - 先减小 `global_batch_size`，再考虑减小模型规模或切到 CPU 验证。

## 12. 读完这份文档后应该理解什么

无论是学生还是 Codex，读完后都应该能回答：

- 如何从零创建项目训练环境？
- 为什么要先检查依赖，再按 `requirements.txt` 安装？
- 如何从 ModelScope 下载 `Harris/SpongeBobPRO` 数据？
- 为什么数据下载本身也是环境验收的一部分？
- 为什么单卡和双卡都应该各做一次启动验证？
- `scripts/run_pretrain_demo.sh` 和 `train/pretrain.py` 各自负责什么？

如果这些问题都能说清楚，并且正式数据上的单卡 / 双卡训练都能成功启动，就说明环境、依赖、数据、训练入口四件事已经真正串起来了。
