# SpongeBob-Pro 预训练快速教程（官方数据版）

目标：在新环境中完成“依赖安装 -> 官方数据下载到 `data/` -> 启动预训练”闭环，不使用自造数据。

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

任何联网命令前先执行：

```bash
eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
```

依赖安装（按优先级）：

```bash
# 1) 内网源（首选）
python -m pip install -i https://artifactory.stepfun-inc.com/artifactory/api/pypi/pypi-public/simple/ \
  --trusted-host artifactory.stepfun-inc.com \
  modelscope swanlab transformers tokenizers numpy tqdm

# 2) 失败时切换清华源
# python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ modelscope swanlab transformers tokenizers numpy tqdm

# 3) 再失败使用官方 PyPI
# python -m pip install modelscope swanlab transformers tokenizers numpy tqdm
```

## 2. 可选：配置 SwanLab

```bash
# .env 中示例
SWANLAB_API_KEY=你的key
```

训练时如果需要上报，先加载环境变量并设置 `USE_SWANLAB=1`。

## 3. 运行预训练 Demo 脚本

```bash
bash scripts/run_pretrain_demo.sh
```

脚本逻辑：
- 数据目录固定为 `data/pretrain_data/`。
- 训练数据文件名统一为 `data/pretrain_data/spongebob_pretrain_512.bin/.meta`。
- 若该文件不存在，自动执行：
  `modelscope download --dataset Harris/SpongeBobPRO ... --local_dir data/pretrain_data`
- 下载后自动重命名为 `spongebob_pretrain_512.bin/.meta`。
- 然后直接调用 `python train/pretrain_without_ddp.py` 开始训练。

## 4. 常用变量

```bash
# 基础运行
DEVICE=cuda:0 EPOCHS=1 BATCH_SIZE=16 bash scripts/run_pretrain_demo.sh

# 显存不足时建议
BATCH_SIZE=8 HIDDEN_SIZE=512 NUM_LAYERS=8 bash scripts/run_pretrain_demo.sh

# 开启 SwanLab
USE_SWANLAB=1 bash scripts/run_pretrain_demo.sh
```

## 5. 成功标志与排障
- 成功标志：日志出现 `Starting training:`，并在训练过程中出现 `Saved checkpoint:`。
- `modelscope: command not found`：先安装 `modelscope`。
- 下载失败：确认先执行了代理命令再重试。
- CUDA OOM：降低 `BATCH_SIZE`，必要时同时降低 `HIDDEN_SIZE` 与 `NUM_LAYERS`。
