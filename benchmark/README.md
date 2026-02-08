# Benchmark 评测模块

## 目录结构

```
benchmark/
├── evaluator.py              # 评测核心模块
├── clue_c3_eval_500.jsonl    # C3 评测数据（500条，固定采样）
├── xcopa_zh_merged.jsonl     # XCOPA 评测数据（600条）
└── README.md                 # 本文档
```

## 使用方法

### 在预训练中自动评测

```bash
# 默认开启评测（每次保存 checkpoint 时自动评测）
python train/pretrain.py

# 禁用评测
python train/pretrain.py --eval_bench 0
```

### 单独评测模型

```python
from transformers import AutoTokenizer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from benchmark.evaluator import run_benchmark

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained('path/to/tokenizer')
model = SpongeBobForCausalLM.from_pretrained('path/to/checkpoint')

# 运行评测
results = run_benchmark(
    model, 
    tokenizer,
    c3_path='benchmark/clue_c3_eval_500.jsonl',
    xcopa_path='benchmark/xcopa_zh_merged.jsonl'
)

print(f"C3: {results['c3_accuracy']:.2%}")
print(f"XCOPA: {results['xcopa_accuracy']:.2%}")
```

## 评测指标

- **C3 Accuracy**: 中文多项选择阅读理解准确率（4选1，随机基线 25%）
- **XCOPA Accuracy**: 中文因果推理准确率（2选1，随机基线 50%）

## 数据说明

- **C3**: 从 CLUE C3 validation split 中固定采样 500 条（seed=42）
- **XCOPA**: 合并 validation (100条) + test (500条) = 600 条

## 特性

✅ 自动化评测（保存 checkpoint 时触发）  
✅ 分布式友好（仅主进程评测）  
✅ SwanLab 曲线记录  
✅ 固定数据集，结果可复现
