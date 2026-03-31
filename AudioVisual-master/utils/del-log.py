import matplotlib.pyplot as plt
import os

path = "sound-spaces/data/models/replica/av_nav/none_against_0_3000_regressor_01"
#path = "sound-spaces/data/models/replica/xcx/none-against"
file_path = os.path.join(path, "train.log")

n = 80000
count = 0
lines = None

with open(file_path, "r") as f:
    lines = f.readlines()
    
with open(file_path ,"w",encoding="utf-8") as f_w:
    for line in lines:
        count += 1
        if count > n:
            f_w.write(line)