#!/usr/bin/env bash
set -euo pipefail

# 教学友好版：支持单卡 / 多卡的最小预训练脚本
#
# 用法：
#   bash scripts/run_pretrain_demo.sh
#
# 常用改法：
#   DEVICE=cpu bash scripts/run_pretrain_demo.sh
#   GLOBAL_BATCH_SIZE=256 bash scripts/run_pretrain_demo.sh
#   NPROC_PER_NODE=4 GLOBAL_BATCH_SIZE=512 bash scripts/run_pretrain_demo.sh
#
# 可选环境变量：
#   DATA_BIN, SAVE_DIR, DEVICE, EPOCHS, GLOBAL_BATCH_SIZE, LEARNING_RATE
#   NPROC_PER_NODE, MASTER_PORT

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1) 训练数据与输出目录
DATA_BIN="${DATA_BIN:-$ROOT/data/pretrain_data/spongebob_pretrain_512.bin}"
SAVE_DIR="${SAVE_DIR:-$ROOT/pretrain_out/demo}"

# 2) 最核心的训练参数
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

# 3) 分布式参数
# NPROC_PER_NODE=1 表示单卡；大于 1 表示单机多卡
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -f "$DATA_BIN" ]]; then
  echo "[error] data file not found: $DATA_BIN"
  exit 1
fi

mkdir -p "$SAVE_DIR"

TRAIN_ARGS=(
  --data_path "$DATA_BIN"
  --save_dir "$SAVE_DIR"
  --epochs "$EPOCHS"
  --global_batch_size "$GLOBAL_BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --from_weight none
  --from_resume 0
  --use_swanlab 0
  --use_compile 0
  --eval_bench 0
)

echo "[info] data_path=$DATA_BIN"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] epochs=$EPOCHS"
echo "[info] global_batch_size=$GLOBAL_BATCH_SIZE"
echo "[info] learning_rate=$LEARNING_RATE"
echo "[info] nproc_per_node=$NPROC_PER_NODE"

if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  echo "[run] single GPU / single process"
  python train/pretrain.py \
    --device "$DEVICE" \
    "${TRAIN_ARGS[@]}"
else
  echo "[run] multi GPU with torchrun"
  echo "[info] each rank batch size = global_batch_size / nproc_per_node"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_port "$MASTER_PORT" \
    train/pretrain.py \
    "${TRAIN_ARGS[@]}"
fi
