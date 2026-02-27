# finnal_exp

本仓库用于 Audio-Visual 导航相关训练与离线 RL 实验。

## 目录结构
- `run.py`: 主入口（按配置分发 `COLLECT / PT / EVAL / TRAIN`）
- `configs/`: 配置文件
- `env/`: 环境封装
- `network/`: 模型定义（foundation + offline RL 各版本）
- `train/`: 训练与数据转换逻辑
- `run/`: 训练/采集执行器
- `tools/`: 独立工具脚本
- `old/`: 归档的旧代码与实验垃圾代码
- `artifacts/experiments/`: 统一存放实验产物（ckpt/loss/历史实验目录）
- `test/`: 测试与 notebook

## 快速使用
默认入口：
```bash
python run.py
```

`run.py` 根据 `config.TASK_CONFIG` 中 `OPEN=True` 的任务执行：
- `COLLECT`
- `PT`
- `EVAL`
- `TRAIN`

请在 `configs/audiogoal.yaml` 中设置对应任务开关与参数。

## 特征提取脚本
已提供独立脚本：
```bash
python tools/extract_foundation_features.py \
  --raw-data-path /path/to/raw_data \
  --ckpt /path/to/foundation_model.pth \
  --to-path /path/to/output_pt \
  --mode offlinetwoframe
```

## Hybrid -> PT -> OfflineRL 训练流程
推荐使用两阶段：

1. 用 Hybrid 模型做 conversion，生成 OfflineRL 训练 PT  
在 `configs/audiogoal.yaml` 中设置：
- `PT.OPEN: True`
- `PT.TYPE: offlinetwoframe`（调用 `LoadLmdb.load_offline_two_frame`）
- `PT.RAW_DATA_PATH`: 原始 pickle 数据目录
- `PT.TO_PATH`: PT 输出目录
- `PT.CKPT`: Hybrid checkpoint
- 可选：`PT.SHARD_SIZE`（默认 10000）
- 可选：`PT.EMBED_BATCH_SIZE`（默认 64，越大越快但更占显存）

运行：
```bash
python run.py
```

2. 用生成的 PT 训练 OfflineRL  
在 `configs/audiogoal.yaml` 中设置：
- `TRAIN.OPEN: True`
- `TRAIN.TYPE: OfflineRL`
- `TRAIN.OFFLINE.train_shard_pattern`: 第一步输出的 PT 目录（如 `media/pt/offline_with_hybrid`）
- `TRAIN.OFFLINE.ONLINE_CKPT`: Hybrid checkpoint
- 其余超参按 `TRAIN.OFFLINE` 配置（`model/lr/batch_size/num_epochs` 等）

运行：
```bash
python run.py
```

提示：
- 训练时如果数据很大，优先使用 `TRAIN.OFFLINE.RANDOM_SHARD_RELOAD` 降低内存压力。
- conversion 和 training 可以分别执行，便于中断续跑与复用 PT。

## LSTM v1_5 流程（PT -> 训练 -> 在线测试 -> Eval）
### 1) 生成 LSTM 离线编码 PT
在 `configs/audiogoal.yaml` 中设置：
- `PT.OPEN: True`
- `PT.TYPE: offlinelstm_v15`
- `PT.RAW_DATA_PATH`: 原始 pickle 数据目录（或 level 文件列表）
- `PT.TO_PATH`: 输出目录
- `PT.CKPT`: Hybrid checkpoint
- 可选：`PT.SEQ_LEN`（默认 5）
- 可选：`PT.SHARD_SIZE`（默认 2000）

运行：
```bash
python run.py
```

### 2) 训练 OfflineRL v1_5（LSTM）
在 `configs/audiogoal.yaml` 中设置：
- `TRAIN.OPEN: True`
- `TRAIN.TYPE: OfflineRL`
- `TRAIN.OFFLINE.model: v1_5`
- `TRAIN.OFFLINE.train_shard_pattern`: 第一步 PT 输出目录
- `TRAIN.OFFLINE.ONLINE_CKPT`: Hybrid checkpoint（用于 online test 时提 embedding）
- 可选：`TRAIN.OFFLINE.lstm_hidden_dim`
- 可选：`TRAIN.OFFLINE.lstm_num_layers`
- 可选：`TRAIN.OFFLINE.lstm_seq_len`

运行：
```bash
python run.py
```

### 3) OnlineTest（v1_5）
训练时会自动走 `OnlineTest.rollout_lstm_v15`（按 `model=v1_5` 分支）进行在线评测。

### 4) Eval（支持整模或独立权重）
在 `configs/audiogoal.yaml` 中设置：
- `EVAL.OPEN: True`
- `EVAL.TYPE: OfflineRL_v1_5`
- 必填：`EVAL.HYBRID_CKPT`, `EVAL.SAC_CKPT`
- 可选独立权重：
  - `EVAL.TEMPORAL_CKPT`
  - `EVAL.ACTOR_CKPT`
  - `EVAL.CRITIC1_CKPT`
  - `EVAL.CRITIC2_CKPT`

运行：
```bash
python run.py
```

更多接口说明见 `docs/INTERFACE.md`。

## 实验一句话记录（防遗忘）
每次运行 `python run.py` 会自动在 `logs/experiment_journal.md` 追加一行记录。

可选地用环境变量传一句备注：
```bash
EXP_NOTE="offlinegreedy telephone baseline lr=5e-5" python run.py
```
