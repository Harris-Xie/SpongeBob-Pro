#!/usr/bin/env bash
set -euo pipefail

# SpongeBob-Pro 预训练 Demo 脚本（支持单机单卡/多卡）
#
# 用法：
#   bash scripts/run_pretrain_demo.sh
#
# 可选环境变量：
#   DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS
#   DTYPE, NUM_WORKERS, LOG_INTERVAL, SAVE_INTERVAL, USE_SWANLAB, SAVE_DIR, DATA_DIR
#   NPROC_PER_NODE(默认auto), MASTER_PORT(默认29500)

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

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-12}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
USE_SWANLAB="${USE_SWANLAB:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
EVAL_BENCH="${EVAL_BENCH:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/tokenizer_15k}"
C3_PATH="${C3_PATH:-$ROOT/benchmark/clue_c3_eval_500.jsonl}"
XCOPA_PATH="${XCOPA_PATH:-$ROOT/benchmark/xcopa_zh_merged.jsonl}"

DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"


mkdir -p "$DATA_DIR" "$SAVE_DIR"

if [[ ! -f "$DATA_BIN" ]]; then
  echo "[error] data bin not found: $DATA_BIN"
  exit 1
fi
if [[ ! -f "$DATA_META" ]]; then
  echo "[error] data meta not found: $DATA_META"
  exit 1
fi

# 自动推断单机进程数：GPU 模式下默认使用全部可见卡；CPU 固定 1 进程
if [[ "$NPROC_PER_NODE" == "auto" ]]; then
  if [[ "$DEVICE" == cuda* ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      CUDA_VISIBLE_DEVICES_CLEAN="${CUDA_VISIBLE_DEVICES// /}"
      IFS=',' read -r -a CUDA_DEVICES <<<"$CUDA_VISIBLE_DEVICES_CLEAN"
      NPROC_PER_NODE="${#CUDA_DEVICES[@]}"
      if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
        NPROC_PER_NODE=1
      fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
      GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -ge 1 ]]; then
        NPROC_PER_NODE="$GPU_COUNT"
      else
        NPROC_PER_NODE=1
      fi
    else
      NPROC_PER_NODE=1
    fi
  else
    NPROC_PER_NODE=1
  fi
fi

if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "[error] invalid NPROC_PER_NODE: $NPROC_PER_NODE"
  exit 1
fi

# =========================
# 2) 启动训练
# =========================
echo "[run] device: $DEVICE"
echo "[run] data: $DATA_BIN"
echo "[run] save_dir: $SAVE_DIR"
echo "[run] epochs=$EPOCHS batch_size=$BATCH_SIZE hidden_size=$HIDDEN_SIZE layers=$NUM_LAYERS lr=$LEARNING_RATE dtype=$DTYPE use_swanlab=$USE_SWANLAB eval_bench=$EVAL_BENCH nproc_per_node=$NPROC_PER_NODE"

TRAIN_ARGS=(
  --data_path "$DATA_BIN"
  --save_dir "$SAVE_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --dtype "$DTYPE"
  --num_workers "$NUM_WORKERS"
  --accumulation_steps 1
  --grad_clip 1.0
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --hidden_size "$HIDDEN_SIZE"
  --num_hidden_layers "$NUM_LAYERS"
  --max_seq_len "$MAX_SEQ_LEN"
  --from_weight none
  --from_resume 0
  --use_swanlab "$USE_SWANLAB"
  --use_compile "$USE_COMPILE"
  --eval_bench "$EVAL_BENCH"
  --eval_interval "$EVAL_INTERVAL"
  --tokenizer_path "$TOKENIZER_PATH"
  --c3_path "$C3_PATH"
  --xcopa_path "$XCOPA_PATH"
)

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  if [[ "$DEVICE" != cuda* ]]; then
    echo "[error] multi-card requires cuda device, got DEVICE=$DEVICE"
    exit 1
  fi
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "[error] torchrun not found in PATH"
    exit 1
  fi
  echo "[run] launch mode: torchrun (single-node multi-gpu)"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    train/pretrain.py \
    "${TRAIN_ARGS[@]}"
else
  echo "[run] launch mode: python (single process)"
  python train/pretrain.py \
    --device "$DEVICE" \
    "${TRAIN_ARGS[@]}"
fi

echo "[done] pretrain demo finished."
