from train import foundation_train
from train import VADE
import torch
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset , DataLoader

def Train(model , config , device = None):
    
    from torch.utils.tensorboard import SummaryWriter
    
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    save_dir = config.EXPERIMENT_CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=config.EXPERIMENT_LOSS_DIR)
    
    train_dataset = VADE(path=config.TRAIN_DATASET)
    val_dataset = VADE(path = config.VAL_DATASET)
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True , num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True , num_workers=8, pin_memory=True)
    trainer = foundation_train(
        model=model,
        Adam=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epoch=config.EPOCH,
        writer=writer,
        device=device,
        save_dir = save_dir
    )
    trainer.train()