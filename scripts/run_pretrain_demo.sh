#!/usr/bin/env bash
set -euo pipefail

# SpongeBob-Pro 预训练 Demo 脚本（官方数据版）
#
# 用法：
#   bash scripts/run_pretrain_demo.sh
#
# 可选环境变量：
#   DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS
#   DTYPE, NUM_WORKERS, LOG_INTERVAL, SAVE_INTERVAL, USE_SWANLAB, SAVE_DIR, DATA_DIR

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false

# 从 .env 读取环境变量（例如 SWANLAB_API_KEY）
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# =========================
# 1) 路径与默认参数
# =========================
DATA_DIR="${DATA_DIR:-$ROOT/data/pretrain_data}"
DATA_BIN="$DATA_DIR/spongebob_pretrain_512.bin"
DATA_META="$DATA_DIR/spongebob_pretrain_512.meta"
SAVE_DIR="${SAVE_DIR:-$ROOT/pretrain_out/demo}"
DOWNLOADED_BIN="$DATA_DIR/SpongeBobPRO_pretrain_512_final.bin"
DOWNLOADED_META="$DATA_DIR/SpongeBobPRO_pretrain_512_final.meta"

if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_DEVICE="cuda:0"
else
  DEFAULT_DEVICE="cpu"
fi

DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-12}"
DTYPE="${DTYPE:-bfloat16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
USE_SWANLAB="${USE_SWANLAB:-0}"

mkdir -p "$DATA_DIR" "$SAVE_DIR"

# =========================
# 2) 启动训练
# =========================
echo "[run] device: $DEVICE"
echo "[run] data: $DATA_BIN"
echo "[run] save_dir: $SAVE_DIR"
echo "[run] epochs=$EPOCHS batch_size=$BATCH_SIZE hidden_size=$HIDDEN_SIZE layers=$NUM_LAYERS lr=$LEARNING_RATE dtype=$DTYPE use_swanlab=$USE_SWANLAB"

python train/pretrain_without_ddp.py \
  --data_path "$DATA_BIN" \
  --save_dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num_workers "$NUM_WORKERS" \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval "$LOG_INTERVAL" \
  --save_interval "$SAVE_INTERVAL" \
  --hidden_size "$HIDDEN_SIZE" \
  --num_hidden_layers "$NUM_LAYERS" \
  --max_seq_len 512 \
  --from_weight none \
  --from_resume 0 \
  --use_swanlab "$USE_SWANLAB" \
  --use_compile 0 \
  --eval_bench 0

echo "[done] pretrain demo finished."
