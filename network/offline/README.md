# OfflineRL Version Notes (`v1_*`)

This file summarizes what each `network/offline/v1_*` version is doing in this repo.

## Version Summary

| Version | Core idea | Input form | Status |
|---|---|---|---|
| `v1` | Basic discrete SAC + CQL | Pre-embedded state vector | Legacy (kept) |
| `v1_2` | `v1` variant with adaptive `beta` logic | Pre-embedded state vector | Legacy experiment (not wired in `run.py`) |
| `v1_3` | Jointly train Hybrid encoder + SAC-CQL | Raw `(audio, rgb, depth)` | Legacy (kept for eval/ablation) |
| `v1_4` | `v1_3` + LSTM temporal encoder | Sequence of raw observations | Legacy (kept) |
| `v1_5` | Offline LSTM temporal SAC-CQL | PT shard sequence `[B, T, D]` | Stable baseline |
| `v1_6` | Transformer temporal SAC-CQL (older impl) | PT shard sequence `[B, T, D]` | Deprecated alias |
| `v1_8` | Causal Transformer temporal SAC-CQL | PT shard sequence `[B, T, D]` | Current Transformer version |

## Deduplication Decision

- Transformer implementation is now unified to `v1_8`.
- `v1_6` is treated as a compatibility alias in `run.py`:
  - If config uses `OFFLINE.model: "v1_6"`, runtime logs a warning and instantiates `v1_8`.
- This removes duplicate Transformer code paths while preserving old config compatibility.

## Current Recommendation

- For sequence-memory OfflineRL: use `OFFLINE.model: "v1_8"`.
- Keep PT generation in sequence mode (`offlinelstm_v15` / `offlinetransformer_v18`) with matching `SEQ_LEN`.
