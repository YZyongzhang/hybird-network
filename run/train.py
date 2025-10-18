from train import foundation_train
from train import VADE
from train import ShardedPTDataset
import torch
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset , DataLoader
import pickle
def Train(model , config , device = None):
    
    from torch.utils.tensorboard import SummaryWriter
    
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-5
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
    with open('loader.pkl' , 'wb') as f:
        pickle.dump(val_loader , f)
    trainer = foundation_train(
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