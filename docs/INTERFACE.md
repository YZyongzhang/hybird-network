# 接口说明

## 1) 主入口接口
文件：`run.py`

### 任务分发
`run.py` 会读取 `get_config()`，并按以下优先级执行第一个 `OPEN=True` 的任务：
1. `TASK_CONFIG.COLLECT`
2. `TASK_CONFIG.PT`
3. `TASK_CONFIG.EVAL`
4. `TASK_CONFIG.TRAIN`

### PT 数据构建（`TASK_CONFIG.PT.TYPE`）
- `HybirdNetwork`: 原始样本转基础 PT（不走 embedding）
- `HybirdNetworkTwoFrame`: 双帧原始拼接 PT
- `OfflineWithHybridPT`: 保存 `(audio, rgbd_pair)` tuple 结构
- `OfflineTwoFrameWithHybridPT`: foundation embedding + 双帧
- `offline`: foundation embedding + 单帧
- `offlinetwoframe`: foundation embedding + 双帧
- `offlinelstm`: foundation embedding + LSTM 序列
- `offlinelstm_by_level`: 读取 level 文件列表的 LSTM 序列
- `offlinelstm_by_level_audio_visual`: 分开保存 audio/visual embedding 序列

### EVAL 类型
- `OfflineRL_v1_3`

### TRAIN 类型
- `HybirdNetworkAudio`
- `HybirdNetwork`
- `OfflineRL`（内部再按 `OFFLINE.model` 分支：`v1/v1_3/v1_4/v2/v4/v5`）

## 2) Foundation 特征提取接口
文件：`tools/extract_foundation_features.py`

## 0) VADE 重构说明
- 兼容入口仍为 `train/VADE.py`（外部导入无需修改）
- 实际实现已拆分为：
  - `train/vade_conversion.py`：`LoadLmdb`（数据转 PT / embedding 提取）
  - `train/vade_datasets.py`：`ShardedPTDataset*`（离线 PT 数据集）
- 已移除当前主流程未使用的 legacy 工具，并归档到：
  - `old/vade_legacy_utils.py`

### 命令
```bash
python tools/extract_foundation_features.py \
  --raw-data-path <raw_data_dir> \
  --ckpt <foundation_ckpt.pth> \
  --to-path <output_dir> \
  --mode <offline|offlinetwoframe>
```

### 参数
- `--raw-data-path`: 原始离线数据目录
- `--ckpt`: foundation checkpoint
- `--to-path`: 输出 `.pt` 分片目录
- `--mode`:
  - `offline`: 单帧 embedding
  - `offlinetwoframe`: 双帧拼接后 embedding

### 输出
按 shard 保存为：
- `offline_rl_shard_0.pt`
- `offline_rl_shard_1.pt`
- ...

字段通常包含：`states`, `next_states`, `actions`, `rewards`, `dones`。

## 3) 归档策略
- 明显实验性、无生产调用路径的代码，移动到 `old/`
- 当前已归档：`old/ai_code_rubbish/`

后续如要继续归档，建议先用 `rg` 检查引用再移动。
