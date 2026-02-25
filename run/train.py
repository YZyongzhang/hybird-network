from train import VADE
from train import (
    ShardedPTDataset,
    ShardedPTDatasetOffline,
    ShardedPTDatasetOfflineBuffer,
    HybridOfflineDataset,
    ChunkedHybridOfflineDataset,
    ChunkAwareBatchSampler,
    ChunkWindowBatchSampler,
)
import torch
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset , DataLoader
import pickle


def Train(model ,trainer , config , device = None , **kwargs):
    if config.TYPE == "OfflineRL":
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        from torch.utils.tensorboard import SummaryWriter
        save_dir = config.EXPERIMENT_CKPT_DIR
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.EXPERIMENT_LOSS_DIR)
        num_workers = int(getattr(config, "NUM_WORKERS", 10))
        use_chunked_lmdb = bool(getattr(config, "USE_CHUNKED_LMDB", False))
        chunk_aware_batch = bool(getattr(config, "CHUNK_AWARE_BATCH", True))
        chunk_window_batch = bool(getattr(config, "CHUNK_WINDOW_BATCH", False))
        chunk_window_size = int(getattr(config, "CHUNK_WINDOW_SIZE", 8))
        chunk_window_log = bool(getattr(config, "CHUNK_WINDOW_LOG", False))
        chunk_window_log_every = int(getattr(config, "CHUNK_WINDOW_LOG_EVERY", 100))
        chunks_per_epoch = int(getattr(config, "CHUNKS_PER_EPOCH", 8))
        shuffle_chunks = bool(getattr(config, "SHUFFLE_CHUNKS", True))
        chunk_replace_ratio = float(getattr(config, "CHUNK_REPLACE_RATIO", 0.2))
        readahead = bool(getattr(config, "LMDB_READAHEAD", True))
        persistent_workers = bool(getattr(config, "PERSISTENT_WORKERS", True))
        prefetch_factor = int(getattr(config, "PREFETCH_FACTOR", 2))

        
        if config.model == "v5":
            assert not config.buffer
            train_dataset = ShardedPTDatasetOffline(train_json=config.train_shard_pattern , attention= True)
        else:
            if config.buffer:
                train_dataset = ShardedPTDatasetOfflineBuffer(train_json=config.train_shard_pattern)
            else:
                if use_chunked_lmdb:
                    if config.model != "v1_3":
                        raise ValueError(
                            f"USE_CHUNKED_LMDB=True is currently supported for model v1_3 only, got {config.model}"
                        )
                    train_dataset = ChunkedHybridOfflineDataset(
                        lmdb_path=config.train_lmdb_path,
                        chunks_per_epoch=chunks_per_epoch,
                        readahead=readahead,
                        shuffle_chunks=shuffle_chunks,
                        log_chunk_loading=bool(getattr(config, "LOG_CHUNK_LOADING", True)),
                        profile_chunk_time=bool(getattr(config, "PROFILE_CHUNK_TIME", False)),
                        chunk_replace_ratio=chunk_replace_ratio,
                    )
                else:
                    train_dataset = HybridOfflineDataset(
                        lmdb_path=config.train_lmdb_path
                    )
        print(train_dataset.__len__())
        if use_chunked_lmdb:
            print(
                f"[Train] chunked_lmdb=True sampler_window={chunk_window_batch} "
                f"window_size={chunk_window_size} chunks_per_epoch={chunks_per_epoch} "
                f"num_workers={num_workers} prefetch={prefetch_factor}"
            )
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": (persistent_workers and num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        if use_chunked_lmdb and chunk_aware_batch:
            if chunk_window_batch:
                batch_sampler = ChunkWindowBatchSampler(
                    dataset=train_dataset,
                    batch_size=config.batch_size,
                    window_chunks=chunk_window_size,
                    drop_last=False,
                    log_window=chunk_window_log,
                    log_every_batches=chunk_window_log_every,
                )
            else:
                batch_sampler = ChunkAwareBatchSampler(
                    dataset=train_dataset,
                    batch_size=config.batch_size,
                    drop_last=False,
                )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                **loader_kwargs,
            )
        if config.SAVE_LOADER:
            with open('train_loader.pkl' , 'wb') as f:
                pickle.dump(train_loader , f)
                
        online_test = kwargs['online_test']
        trainer = trainer(
            dataset = train_dataset,
            dataloader = train_loader , 
            sac_model=model,
            device=device,
            writer=writer,
            online_test = online_test,
            online_test_epoch = config.online_test_epoch,
            batch_size = config.batch_size,
            epoch = config.num_epochs,
            save_dir = save_dir,
            config = config
        )
        trainer.train()
        
    else:
        from torch.utils.tensorboard import SummaryWriter
        
        if device is not None:
            device = device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lr = 1e-6
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        save_dir = config.EXPERIMENT_CKPT_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=config.EXPERIMENT_LOSS_DIR)
        
        train_dataset = ShardedPTDataset(shard_pattern=config.train_shard_pattern)
        val_dataset = ShardedPTDataset(shard_pattern=config.val_shard_pattern)
        print(train_dataset.__len__())
        print(val_dataset.__len__())
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True ,  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True , pin_memory=True)
        if config.SAVE_LOADER:
            with open('val_loader.pkl' , 'wb') as f:
                pickle.dump(val_loader , f)
            with open('train_loader.pkl' , 'wb') as f:
                pickle.dump(train_loader , f)
        trainer = trainer(
            model=model,
            Adam=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=config.EPOCH,
            writer=writer,
            device=device,
            save_dir = save_dir,
            train_size=train_dataset.__len__(),
            val_size=val_dataset.__len__()
        )
        trainer.train()
