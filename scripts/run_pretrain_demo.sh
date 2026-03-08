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
#   HIDDEN_SIZE=768 NUM_ATTENTION_HEADS=12 HEAD_SIZE=80 bash scripts/run_pretrain_demo.sh
#
# 可选环境变量：
#   DATA_BIN, SAVE_DIR, DEVICE, EPOCHS, GLOBAL_BATCH_SIZE, LEARNING_RATE
#   LOG_INTERVAL, SAVE_INTERVAL, EVAL_INTERVAL
#   HIDDEN_SIZE, NUM_HIDDEN_LAYERS, NUM_ATTENTION_HEADS, HEAD_SIZE
#   NUM_KEY_VALUE_HEADS, INTERMEDIATE_SIZE, VOCAB_SIZE
#   MAX_SEQ_LEN, MAX_POSITION_EMBEDDINGS, ROPE_THETA, HIDDEN_ACT, DROPOUT
#   NPROC_PER_NODE, MASTER_PORT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# 0）环境变量
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "[info] loaded env file: $ENV_FILE"
fi

# 1) 训练数据与输出目录
DATA_BIN="${DATA_BIN:-$ROOT/data/pretrain_data/spongebob_pretrain_512.bin}"
SAVE_DIR="${SAVE_DIR:-$ROOT/pretrain_out/demo}"

# 2) 最核心的训练参数
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-12}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-12}"
HEAD_SIZE="${HEAD_SIZE:-64}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
VOCAB_SIZE="${VOCAB_SIZE:-15000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-32768}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
HIDDEN_ACT="${HIDDEN_ACT:-silu}"
DROPOUT="${DROPOUT:-0.0}"

# 3) 分布式参数
# NPROC_PER_NODE=1 表示单卡；大于 1 表示单机多卡
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 4）评测/swanlab记录参数
EVAL_BENCH="${EVAL_BENCH:-1}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SpongeBob-Pretrain-New}"

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
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --eval_interval "$EVAL_INTERVAL"
  --hidden_size "$HIDDEN_SIZE"
  --num_hidden_layers "$NUM_HIDDEN_LAYERS"
  --num_attention_heads "$NUM_ATTENTION_HEADS"
  --num_key_value_heads "$NUM_KEY_VALUE_HEADS"
  --intermediate_size "$INTERMEDIATE_SIZE"
  --vocab_size "$VOCAB_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
  --max_position_embeddings "$MAX_POSITION_EMBEDDINGS"
  --rope_theta "$ROPE_THETA"
  --hidden_act "$HIDDEN_ACT"
  --dropout "$DROPOUT"
  --from_weight none
  --from_resume 0
  --use_swanlab "$USE_SWANLAB"
  --use_compile 0
  --eval_bench "$EVAL_BENCH"
  --swanlab_project "$SWANLAB_PROJECT"
  --head_size "$HEAD_SIZE"
)

echo "[info] data_path=$DATA_BIN"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] epochs=$EPOCHS"
echo "[info] global_batch_size=$GLOBAL_BATCH_SIZE"
echo "[info] learning_rate=$LEARNING_RATE"
echo "[info] log_interval=$LOG_INTERVAL"
echo "[info] save_interval=$SAVE_INTERVAL"
echo "[info] eval_interval=$EVAL_INTERVAL"
echo "[info] hidden_size=$HIDDEN_SIZE"
echo "[info] num_hidden_layers=$NUM_HIDDEN_LAYERS"
echo "[info] num_attention_heads=$NUM_ATTENTION_HEADS"
echo "[info] head_size=$HEAD_SIZE"
echo "[info] num_key_value_heads=$NUM_KEY_VALUE_HEADS"
echo "[info] intermediate_size=$INTERMEDIATE_SIZE"
echo "[info] vocab_size=$VOCAB_SIZE"
echo "[info] max_seq_len=$MAX_SEQ_LEN"
echo "[info] max_position_embeddings=$MAX_POSITION_EMBEDDINGS"
echo "[info] rope_theta=$ROPE_THETA"
echo "[info] hidden_act=$HIDDEN_ACT"
echo "[info] dropout=$DROPOUT"
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
