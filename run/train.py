from train import (
    ShardedPTDataset,
    ShardedPTDatasetOffline,
    RandomReloadShardedPTDatasetOffline,
    ShardedPTDatasetOfflineBuffer,
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
        persistent_workers = bool(getattr(config, "PERSISTENT_WORKERS", True))
        prefetch_factor = int(getattr(config, "PREFETCH_FACTOR", 2))
        random_shard_reload = bool(getattr(config, "RANDOM_SHARD_RELOAD", False))
        random_shards_per_epoch = int(getattr(config, "RANDOM_SHARDS_PER_EPOCH", 8))
        reload_shards_every_epochs = int(getattr(config, "RELOAD_SHARDS_EVERY_EPOCHS", 10))
        random_shard_seed = getattr(config, "RANDOM_SHARD_SEED", None)

        
        if config.model == "v5":
            assert not config.buffer
            train_dataset = ShardedPTDatasetOffline(
                train_shard_dir=config.train_shard_pattern,
                attention=True,
            )
        else:
            if config.buffer:
                train_dataset = ShardedPTDatasetOfflineBuffer(train_json=config.train_shard_pattern)
            else:
                if random_shard_reload:
                    train_dataset = RandomReloadShardedPTDatasetOffline(
                        train_shard_dir=config.train_shard_pattern,
                        attention=False,
                        shards_per_epoch=random_shards_per_epoch,
                        reload_every_epochs=reload_shards_every_epochs,
                        seed=random_shard_seed,
                    )
                else:
                    train_dataset = ShardedPTDatasetOffline(
                        train_shard_dir=config.train_shard_pattern,
                        attention=False,
                    )
        print(train_dataset.__len__())
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": (persistent_workers and num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

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
