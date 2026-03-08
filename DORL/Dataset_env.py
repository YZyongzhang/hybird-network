from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


TransitionLike = Union[Dict[str, Any], Sequence[Any]]


@dataclass
class Transition:
    state: Any
    next_state: Any
    reward: float
    done: bool
    action: Any


class RLEnv:
    """
    Dataset-driven RL env.

    每个 episode 是一个 transition 列表:
    [
        (state, next_state, reward, done, action),
        ...
    ]
    或者 dict 形式:
    {
        "state": ...,
        "next_state": ...,
        "reward": ...,
        "done": ...,
        "action": ...
    }
    """

    def __init__(
        self,
        episodes: Optional[Sequence[Sequence[TransitionLike]]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        *,
        random_episode: bool = False,
        mismatch_reward: float = -10.0,
        strict_done_from_data: bool = False,
    ) -> None:
        if episodes is None and dataset_path is None:
            raise ValueError("episodes 和 dataset_path 至少提供一个")

        loaded_episodes = episodes
        if loaded_episodes is None and dataset_path is not None:
            loaded_episodes = self._load_dataset(dataset_path)

        self.episodes: List[List[Transition]] = self._normalize_episodes(loaded_episodes or [])
        if not self.episodes:
            raise ValueError("episodes 为空，无法构建环境")

        self.random_episode = random_episode
        self.mismatch_reward = float(mismatch_reward)
        self.strict_done_from_data = strict_done_from_data

        self.episode_idx = -1
        self.step_idx = 0
        self._active_episode: List[Transition] = []
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._state_dtype: Any = np.float32
        self._last_state: Any = None

    def reset(self, episode_idx: Optional[int] = None) -> Any:
        """
        重置到某个 episode 并返回首条 transition 的 state。
        """
        if episode_idx is not None:
            if episode_idx < 0 or episode_idx >= len(self.episodes):
                raise IndexError(f"episode_idx 越界: {episode_idx}")
            self.episode_idx = episode_idx
        else:
            if self.random_episode:
                self.episode_idx = random.randrange(len(self.episodes))
            else:
                self.episode_idx = (self.episode_idx + 1) % len(self.episodes)

        self._active_episode = self.episodes[self.episode_idx]
        if not self._active_episode:
            raise ValueError(f"episode {self.episode_idx} 为空")

        self.step_idx = 0
        first_state = self._active_episode[0].state
        self._state_shape, self._state_dtype = self._infer_state_meta(first_state)
        self._last_state = first_state
        return first_state

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        规则:
        - action == 数据集当前 transition.action:
          返回 transition.next_state, transition.reward, transition.done
        - action != transition.action:
          返回 zero_state, -10(可配), done=True
        """
        if not self._active_episode:
            raise RuntimeError("请先调用 reset()")

        if self.step_idx >= len(self._active_episode):
            # episode 已结束，按 mismatch 处理
            return self._zero_state(), self.mismatch_reward, True, {
                "error": "episode_already_done",
                "episode_idx": self.episode_idx,
                "step_idx": self.step_idx,
            }

        transition = self._active_episode[self.step_idx]
        expected_action = transition.action

        if self._action_equal(action, expected_action):
            next_state = transition.next_state
            reward = float(transition.reward)
            if self.strict_done_from_data:
                done = bool(transition.done)
            else:
                done = bool(transition.done) or (self.step_idx >= len(self._active_episode) - 1)

            self.step_idx += 1
            self._last_state = next_state
            info = {
                "matched": True,
                "episode_idx": self.episode_idx,
                "step_idx": self.step_idx - 1,
                "expected_action": expected_action,
            }
            return next_state, reward, done, info

        zero_state = self._zero_state()
        info = {
            "matched": False,
            "episode_idx": self.episode_idx,
            "step_idx": self.step_idx,
            "expected_action": expected_action,
            "received_action": action,
        }
        return zero_state, self.mismatch_reward, True, info

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def _normalize_episodes(self, episodes: Sequence[Sequence[TransitionLike]]) -> List[List[Transition]]:
        normalized: List[List[Transition]] = []
        for ep_idx, ep in enumerate(episodes):
            parsed_ep: List[Transition] = []
            for tr_idx, tr in enumerate(ep):
                parsed_ep.append(self._to_transition(tr, ep_idx, tr_idx))
            normalized.append(parsed_ep)
        return normalized

    def _to_transition(self, tr: TransitionLike, ep_idx: int, tr_idx: int) -> Transition:
        if isinstance(tr, dict):
            if "next_state" in tr:
                next_state = tr["next_state"]
            elif "nextstate" in tr:
                next_state = tr["nextstate"]
            else:
                raise KeyError(f"episode {ep_idx} transition {tr_idx} 缺失 next_state/nextstate")

            if "state" not in tr or "reward" not in tr or "done" not in tr or "action" not in tr:
                raise KeyError(f"episode {ep_idx} transition {tr_idx} 缺失必要字段")

            return Transition(
                state=tr["state"],
                next_state=next_state,
                reward=float(tr["reward"]),
                done=bool(tr["done"]),
                action=tr["action"],
            )

        if not isinstance(tr, (list, tuple)) or len(tr) < 5:
            raise ValueError(f"episode {ep_idx} transition {tr_idx} 格式错误，应为 5 元组")

        state, next_state, reward, done, action = tr[:5]
        return Transition(
            state=state,
            next_state=next_state,
            reward=float(reward),
            done=bool(done),
            action=action,
        )

    def _infer_state_meta(self, state: Any) -> Tuple[Optional[Tuple[int, ...]], Any]:
        arr = self._as_array_or_none(state)
        if arr is None:
            return None, np.float32
        return arr.shape, arr.dtype

    def _zero_state(self) -> Any:
        # 优先按 state 的 shape/dtype 构造 0；无法推断时返回标量 0.0
        if self._state_shape is None:
            arr = self._as_array_or_none(self._last_state)
            if arr is not None:
                return np.zeros_like(arr)
            return 0.0
        return np.zeros(self._state_shape, dtype=self._state_dtype)

    @staticmethod
    def _as_array_or_none(value: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(value)
            if arr.dtype == np.dtype("O"):
                return None
            return arr
        except Exception:
            return None

    @staticmethod
    def _action_equal(a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            try:
                return bool(np.array_equal(np.asarray(a), np.asarray(b)))
            except Exception:
                return False
        return a == b

    @staticmethod
    def _load_dataset(dataset_path: Union[str, Path]) -> Sequence[Sequence[TransitionLike]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"dataset 文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        if suffix == ".pt":
            try:
                import torch  # type: ignore
            except Exception as exc:
                raise ImportError("加载 .pt 需要 torch") from exc
            data = torch.load(path, map_location="cpu")
            return data

        raise ValueError(f"不支持的 dataset 格式: {suffix}")

    @classmethod
    def from_pickle_roots(
        cls,
        roots: Sequence[Union[str, Path]],
        *,
        random_episode: bool = False,
        mismatch_reward: float = -10.0,
        strict_done_from_data: bool = False,
        shuffle_files: bool = False,
        limit_files: Optional[int] = None,
    ) -> "RLEnv":
        """
        从原始离线 pickle 目录构建 RLEnv。
        兼容 train.vade_conversion 里使用的数据格式:
        - data["obs"]: 长度 T+1
        - data["action_id"], data["reward"], data["done"]: 长度 T（可能嵌套）
        """
        files = _collect_pickle_files(roots)
        if shuffle_files:
            random.shuffle(files)
        if limit_files is not None:
            files = files[: int(limit_files)]

        episodes = []
        for fp in files:
            with open(fp, "rb") as f:
                data = pickle.load(f)
            ep = _episode_from_raw_pickle(data)
            if ep:
                episodes.append(ep)

        return cls(
            episodes=episodes,
            random_episode=random_episode,
            mismatch_reward=mismatch_reward,
            strict_done_from_data=strict_done_from_data,
        )

    @classmethod
    def from_dorl_pt_dir(
        cls,
        root: Union[str, Path],
        *,
        pattern: str = "dorl_episode_shard_*.pt",
        random_episode: bool = False,
        mismatch_reward: float = -10.0,
        strict_done_from_data: bool = False,
        limit_shards: Optional[int] = None,
    ) -> "RLEnv":
        """
        从 Generate_DORL_PT 产出的 shard 目录加载 episode。
        """
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise ImportError("from_dorl_pt_dir 需要 torch") from exc

        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"DORL pt 目录不存在: {root_path}")

        shard_files = sorted(root_path.glob(pattern))
        if limit_shards is not None:
            shard_files = shard_files[: int(limit_shards)]
        if not shard_files:
            raise FileNotFoundError(f"未找到 shard: {root_path}/{pattern}")

        episodes: List[Any] = []
        for sf in shard_files:
            data = torch.load(sf, map_location="cpu")
            if isinstance(data, dict) and "episodes" in data:
                data = data["episodes"]
            if isinstance(data, list):
                episodes.extend(data)
            else:
                raise ValueError(f"shard 数据格式错误: {sf}")

        return cls(
            episodes=episodes,
            random_episode=random_episode,
            mismatch_reward=mismatch_reward,
            strict_done_from_data=strict_done_from_data,
        )


def _flatten_sequence(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for v in value:
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend(_flatten_sequence(v))
            else:
                out.append(v)
        return out
    return [value]


def _collect_pickle_files(roots: Sequence[Union[str, Path]]) -> List[str]:
    files: List[str] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # 兼容传入的是“文件列表 pickle”
        if root_path.is_file():
            if root_path.suffix.lower() in {".pkl", ".pickle"}:
                with open(root_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, (list, tuple)):
                    for p in loaded:
                        if isinstance(p, (str, os.PathLike)) and Path(p).is_file():
                            files.append(os.fspath(p))
            continue

        # 目录模式: 递归收集 pickle，忽略 a.md
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            if p.name == "a.md":
                continue
            if p.suffix.lower() in {".pkl", ".pickle"} or p.suffix == "":
                files.append(str(p))
    return sorted(set(files))


def _episode_from_raw_pickle(data: Dict[str, Any]) -> List[Tuple[Any, Any, float, bool, Any]]:
    obs = data.get("obs", [])
    action_id = _flatten_sequence(data.get("action_id", []))
    rewards = _flatten_sequence(data.get("reward", []))
    dones = _flatten_sequence(data.get("done", []))
    n = min(len(action_id), len(rewards), len(dones), max(0, len(obs) - 1))

    episode: List[Tuple[Any, Any, float, bool, Any]] = []
    for i in range(n):
        state = obs[i]
        next_state = obs[i + 1]
        reward = float(rewards[i])
        done = bool(dones[i])
        action = action_id[i]
        episode.append((state, next_state, reward, done, action))
    return episode


def build_env_offline_15_multiple(
    *,
    root: Union[str, Path] = "dataset-data3/dataset/pickle/offline(1.5_muti)/level3",
    random_episode: bool = False,
    mismatch_reward: float = -10.0,
    strict_done_from_data: bool = False,
    shuffle_files: bool = False,
    limit_files: Optional[int] = None,
) -> RLEnv:
    """
    构建 offline(1.5_muti)/level3 的 DatasetEnv。
    """
    return RLEnv.from_pickle_roots(
        roots=[root],
        random_episode=random_episode,
        mismatch_reward=mismatch_reward,
        strict_done_from_data=strict_done_from_data,
        shuffle_files=shuffle_files,
        limit_files=limit_files,
    )


def build_env_offline_collided(
    *,
    root: Union[str, Path] = "dataset-data3/dataset/pickle/offline(muti_collided)",
    random_episode: bool = False,
    mismatch_reward: float = -10.0,
    strict_done_from_data: bool = False,
    shuffle_files: bool = False,
    limit_files: Optional[int] = None,
) -> RLEnv:
    """
    构建 offline(muti_collided) 的 DatasetEnv。
    """
    return RLEnv.from_pickle_roots(
        roots=[root],
        random_episode=random_episode,
        mismatch_reward=mismatch_reward,
        strict_done_from_data=strict_done_from_data,
        shuffle_files=shuffle_files,
        limit_files=limit_files,
    )


def build_env_offline_15_and_collided(
    *,
    offline_15_root: Union[str, Path] = "dataset-data3/dataset/pickle/offline(1.5_muti)/level3",
    collided_root: Union[str, Path] = "dataset-data3/dataset/pickle/offline(muti_collided)",
    random_episode: bool = False,
    mismatch_reward: float = -10.0,
    strict_done_from_data: bool = False,
    shuffle_files: bool = False,
    limit_files: Optional[int] = None,
) -> RLEnv:
    """
    合并加载 offline(1.5_muti) + offline(muti_collided)。
    """
    return RLEnv.from_pickle_roots(
        roots=[offline_15_root, collided_root],
        random_episode=random_episode,
        mismatch_reward=mismatch_reward,
        strict_done_from_data=strict_done_from_data,
        shuffle_files=shuffle_files,
        limit_files=limit_files,
    )


def build_env_from_dorl_pt(
    *,
    root: Union[str, Path] = "",
    pattern: str = "dorl_episode_shard_*.pt",
    random_episode: bool = False,
    mismatch_reward: float = -10.0,
    strict_done_from_data: bool = False,
    limit_shards: Optional[int] = None,
) -> RLEnv:
    return RLEnv.from_dorl_pt_dir(
        root=root,
        pattern=pattern,
        random_episode=random_episode,
        mismatch_reward=mismatch_reward,
        strict_done_from_data=strict_done_from_data,
        limit_shards=limit_shards,
    )
