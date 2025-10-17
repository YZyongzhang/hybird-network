import sys
 
from module import *
from network.foundation_model import Network
from train.VADE import VADE
path = './experiment/dataset/test_label_val'
train_dataset = VADE(path=path)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True )
_0, _1, _2, _3 = 0, 0, 0, 0

for batch in tqdm(train_loader):
    batch_visual, batch_audio, batch_action = batch

    for a in batch_action:
        if a == 0:
            _0 += 1
        elif a == 1:
            _1 += 1
        elif a == 2:
            _2 += 1
        elif a == 3:
            _3 += 1

print(f"label=0: {_0}")
print(f"label=1: {_1}")
print(f"label=2: {_2}")
print(f"label=3: {_3}")
    
    