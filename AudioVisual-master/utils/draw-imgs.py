import matplotlib.pyplot as plt
import os

# path = "sound-spaces/data/models/replica/whc/audiogoal_depth_whc_none_against_reg01"
# path = "sound-spaces/data/models/replica/av_nav/2022-02-12-01-22-01-classifier_layers_2"
#path = "sound-spaces/data/models/replica/xcx/none-against"
path = "./data/models/mp3d/xcx/test"
file_path = os.path.join(path, "train.log")

keys = ["z_max"]
value = {}
for key in keys:
    value[key] = []

freq = 1  # 用来平滑曲线，每freq个数据点取均值画一个图像点
count = 0
sum = 0

with open(file_path) as f:
    lines = f.readlines()
    for line in lines:
        for key in keys:
            if key in line:
                pos = line.find(key)
                if line[pos - 1] == ' ':  # 有个softspl需要排除
                    count += 1
                    sum += float(line[pos + 2 + len(key):])
                    if (count % freq == 0):
                        value[key].append(sum / freq)
                        sum = 0

x = []
for i in range(len(value[keys[0]])):
    x.append(i)

for key in keys:
    plt.plot(x, value[key], label=key)
plt.legend()
plt.show()
plt.savefig(f'{path}/image_{keys[0]}.jpg', dpi=400, bbox_inches='tight')
plt.cla()
print("done")
