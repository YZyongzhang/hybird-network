from torch.utils.data import Dataset
import os
import glob
import bisect
import gc
import random
import torch

class ShardedPTDataset(Dataset):
    def __init__(self, shard_pattern, preload=True):
        super().__init__()
        self.shard_files = []
        for pattern in shard_pattern:
            self.shard_files.extend(sorted(glob.glob(pattern))[:7])
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"
        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan shard id {shard_id}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]

        rgb = data["rgb"][local_idx]
        depth = data['depth'][local_idx]
        audio = data["audios"][local_idx]
        action = data["actions"][local_idx]
        # action_id = data["action_ids"][local_idx]
        # angle = data['angles'][local_idx]
        # consistency_action = data["consistency_actions"][local_idx]
        std_audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        return  std_audio, rgb , depth  ,action
      
class ShardedPTHybridDataset(Dataset):
    def __init__(self, shard_pattern, preload=True, normalize_audio=True):
        super().__init__()
        self.shard_files = []
        for pattern in shard_pattern:
            self.shard_files.extend(sorted(glob.glob(pattern)))
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"
        self.preload = preload
        self.normalize_audio = normalize_audio
        self.shards = []
        self.shard_sizes = []
        self.index_map = []

        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan hybrid shard id {shard_id}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)
            for i in range(size):
                self.index_map.append((shard_id, i))
            if preload:
                self.shards.append(data)
            else:
                self.shards.append(None)

        self.total_size = sum(self.shard_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        shard_id, local_idx = self.index_map[index]

        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]

        rgb = data["rgb"][local_idx]
        depth = data["depth"][local_idx]
        audio = data["audios"][local_idx]
        action = data["actions"][local_idx]
        action_id = data["action_ids"][local_idx]
        angle = data["angles"][local_idx]
        sound_id = data["sound_ids"][local_idx]
        pose = data["pose"][local_idx]
        ego_map = data["ego_map"][local_idx]
        collision = data["collision"][local_idx]
        consistency_action = data["consistency_actions"][local_idx]
        if self.normalize_audio:
            audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        return audio, rgb, depth, action, action_id, angle, sound_id, pose, ego_map, collision, consistency_action

class ShardedPTDatasetOffline(Dataset):
    def __init__(self, train_shard_dir, attention=False, preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = []
        self.use_attention = attention
        self.source_dirs = self.get_files(train_shard_dir)
        # for pattern in shard_pattern:
        #     lists_ = glob.glob(pattern)
        #     random.shuffle(lists_)
        #     import pdb;pdb.set_trace()
        #     self.shard_files.extend(lists_[:40])
        assert len(self.shard_files) > 0, f"No shards found at {train_shard_dir}"
        print(
            f"[ShardedPTDatasetOffline] found {len(self.shard_files)} shards from {len(self.source_dirs)} directories"
        )

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"[ShardedPTDatasetOffline] loading shard {shard_id}: {shard_file}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)
            print(f"[ShardedPTDatasetOffline] shard {shard_id} samples={size}")

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)
        print(f"[ShardedPTDatasetOffline] total samples: {self.total_size}")

    def get_files(self, train_shard_dir):
        if isinstance(train_shard_dir, (str, os.PathLike)):
            dirs = [train_shard_dir]
        else:
            try:
                dirs = list(train_shard_dir)
            except TypeError as exc:
                raise ValueError(
                    "train_shard_pattern must be a directory path or a sequence of directory paths"
                ) from exc

        if not dirs:
            raise ValueError("train_shard_pattern must contain at least one directory path")

        normalized_dirs = []
        for dir_path in dirs:
            dir_str = os.fspath(dir_path)
            if not os.path.isdir(dir_str):
                raise ValueError(
                    f"train_shard_pattern must be a directory path, got: {dir_str}"
                )
            files = sorted(glob.glob(os.path.join(dir_str, "offline_rl_shard_*.pt")))
            if not files:
                files = sorted(glob.glob(os.path.join(dir_str, "*.pt")))
            self.shard_files.extend(files)
            normalized_dirs.append(dir_str)
            print(
                f"[ShardedPTDatasetOffline] shard glob matched {len(files)} files in {dir_str}"
            )

        return normalized_dirs

    def replay(self):
        pass

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]


        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data['next_states_audio'][local_idx]
            next_states_visual_audio = data['next_states_visual_audio'][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (states_audio , states_visual_audio), (next_states_audio , next_states_visual_audio) , action, reward, done
        else:
            # 取出一个 transition
            state       = data["states"][local_idx]
            next_state  = data["next_states"][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            # state = state.squeeze(0)
            # next_state = next_state.squeeze(0)
            return state, next_state , action, reward, done

class RandomReloadShardedPTDatasetOffline(Dataset):
    def __init__(
        self,
        train_shard_dir,
        attention=False,
        shards_per_epoch=8,
        reload_every_epochs=10,
        seed=None,
    ):
        super().__init__()
        self.shard_files = []
        self.use_attention = attention
        self._rng = random.Random(seed) if seed is not None else random
        self.source_dirs = self.get_files(train_shard_dir)
        assert len(self.shard_files) > 0, f"No shards found at {train_shard_dir}"

        self.shards_per_epoch = max(1, int(shards_per_epoch))
        self.reload_every_epochs = max(1, int(reload_every_epochs))
        self._epoch_counter = 0

        self.active_shard_files = []
        self.active_shards = []
        self.cumulative_sizes = []
        self.total_size = 0
        self._reload_shards(initial=True)

    def get_files(self, train_shard_dir):
        if isinstance(train_shard_dir, (str, os.PathLike)):
            dirs = [train_shard_dir]
        else:
            try:
                dirs = list(train_shard_dir)
            except TypeError as exc:
                raise ValueError(
                    "train_shard_pattern must be a directory path or a sequence of directory paths"
                ) from exc

        if not dirs:
            raise ValueError("train_shard_pattern must contain at least one directory path")

        normalized_dirs = []
        for dir_path in dirs:
            dir_str = os.fspath(dir_path)
            if not os.path.isdir(dir_str):
                raise ValueError(
                    f"train_shard_pattern must be a directory path, got: {dir_str}"
                )
            files = sorted(glob.glob(os.path.join(dir_str, "offline_rl_shard_*.pt")))
            if not files:
                files = sorted(glob.glob(os.path.join(dir_str, "*.pt")))
            self.shard_files.extend(files)
            normalized_dirs.append(dir_str)
        return normalized_dirs

    def _rng_shuffle(self, values):
        if hasattr(self._rng, "shuffle"):
            self._rng.shuffle(values)
        else:
            random.shuffle(values)

    def _rng_sample(self, values, k):
        if hasattr(self._rng, "sample"):
            return self._rng.sample(values, k)
        return random.sample(values, k)

    def _reload_shards(self, initial=False):
        target = min(self.shards_per_epoch, len(self.shard_files))
        if target == len(self.shard_files):
            selected = list(self.shard_files)
        else:
            selected = self._rng_sample(self.shard_files, target)

        self.active_shard_files = selected
        # 先释放上一轮缓存，避免 reload 时内存峰值叠加
        old_shards = self.active_shards
        self.active_shards = []
        self.cumulative_sizes = []
        if old_shards:
            old_shards.clear()
            del old_shards
            gc.collect()

        total = 0
        for active_id, shard_file in enumerate(self.active_shard_files):
            print(
                f"[RandomReloadShardedPTDatasetOffline] loading active shard "
                f"{active_id+1}/{len(self.active_shard_files)}: {shard_file}"
            )
            data = torch.load(shard_file, map_location="cpu")
            self.active_shards.append(data)
            size = len(data["actions"])
            total += size
            self.cumulative_sizes.append(total)
            print(
                f"[RandomReloadShardedPTDatasetOffline] active shard {active_id} samples={size}"
            )

        self.total_size = total
        phase = "initial" if initial else "reload"
        print(
            f"[RandomReloadShardedPTDatasetOffline] {phase}: "
            f"loaded_shards={len(self.active_shard_files)} total_samples={self.total_size}"
        )

    def on_epoch_end(self):
        self._epoch_counter += 1
        if self._epoch_counter % self.reload_every_epochs == 0:
            self._reload_shards(initial=False)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        if index < 0:
            index += self.total_size
        if index < 0 or index >= self.total_size:
            raise IndexError(index)
        active_id = bisect.bisect_right(self.cumulative_sizes, index)
        prev_end = 0 if active_id == 0 else self.cumulative_sizes[active_id - 1]
        local_idx = index - prev_end
        data = self.active_shards[active_id]

        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data["next_states_audio"][local_idx]
            next_states_visual_audio = data["next_states_visual_audio"][local_idx]
            action = data["actions"][local_idx]
            reward = data["rewards"][local_idx]
            done = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (
                (states_audio, states_visual_audio),
                (next_states_audio, next_states_visual_audio),
                action,
                reward,
                done,
            )

        state = data["states"][local_idx]
        next_state = data["next_states"][local_idx]
        action = data["actions"][local_idx]
        reward = data["rewards"][local_idx]
        done = data["dones"][local_idx]
        return state, next_state, action, reward, done
        
class ShardedPTDatasetOfflineBuffer(Dataset):
    def __init__(self, train_json,  attention = False , preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        # self.shard_files = []
        self.use_attention = attention
        self.get_file_buffer(train_json)
        # for pattern in shard_pattern:
        #     lists_ = glob.glob(pattern)
        #     random.shuffle(lists_)
        #     import pdb;pdb.set_trace()
        #     self.shard_files.extend(lists_[:40])
        # assert len(self.shard_files) > 0, f"No shards found at {train_json}"

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_greedy_map = []    # 全局 index → (shard_id, local_index)
        self.index_1_map = []
        self.index_2_map = []

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.buffer_greedy):
            print(f"scan shard id {shard_id} and {shard_file} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_greedy_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位


        for shard_id, shard_file in enumerate(self.buffer_1):
            print(f"scan shard id {shard_id} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_1_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        for shard_id, shard_file in enumerate(self.buffer_2):
            print(f"scan shard id {shard_id} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_2_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.index_map = self.index_greedy_map
        self.total_size = sum(self.shard_sizes)
        print(f"Total samples: {self.total_size}")

    def get_files(self , train_json):
        import json
        files = []
        with open(train_json , 'r') as f:
            train_json_data = json.load(f) 
        path = train_json_data['path']
        split = train_json_data['split']
        for beta in path.keys():
            for distance in path[beta].keys():
                print(path[beta][distance])
                lists_1 = glob.glob(path[beta][distance])
                random.shuffle(lists_1)
                files.extend(lists_1[:split[beta][distance]])
                # self.shard_files.extend(lists_1)
        return files
    def get_file_buffer(self , train_json):
        import json
        with open(train_json , 'r') as f:
            train_json_data = json.load(f)
        self.buffer_greedy = self.get_files(train_json=train_json_data['greedy'])
        self.buffer_1 = self.get_files(train_json=train_json_data['1.5'])
        self.buffer_2 = self.get_files(train_json=train_json_data['2.0'])

    def replay(self , logger):
        # 每一次replay greedy减少10k。相应的1.5和2.0各加5k
        # 当greedy减少到50k时 ， 不断sample1.5和2.0，仍确保1.5和2.0始终总体占据50k
        logger.info(f"开始进行replay buffer")
        random.shuffle(self.index_greedy_map)
        random.shuffle(self.index_1_map)
        random.shuffle(self.index_2_map)
        logger.info(f"打乱所有的列表,目前buffer长度{len(self.index_map)}")
        self.index_map = self.index_map[10000:] # 去除前10kgreedy数据
        cache_1_map = self.index_1_map[:5000]
        self.index_1_map = self.index_1_map[5000:]
        cache_2_map = self.index_2_map[:5000]
        self.index_2_map = self.index_2_map[5000:]
        logger.info(f"cache 1 map {len(cache_1_map)}")
        self.index_map.extend(cache_1_map)
        self.index_map.extend(cache_2_map)
        logger.info(f"打乱所有的列表,目前buffer长度{len(self.index_map)} , 1 map {len(self.index_1_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]


        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data['next_states_audio'][local_idx]
            next_states_visual_audio = data['next_states_visual_audio'][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (states_audio , states_visual_audio), (next_states_audio , next_states_visual_audio) , action, reward, done
        else:
            # 取出一个 transition
            state       = data["states"][local_idx]
            next_state  = data["next_states"][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            state = state.squeeze(0)
            next_state = next_state.squeeze(0)

            return state, next_state , action, reward, done
