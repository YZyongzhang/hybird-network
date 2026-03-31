import matplotlib.pyplot as plt
import os

# path = "sound-spaces/data/models/replica/whc/audiogoal_depth_whc_none_against_entropy"
path = "sound-spaces/data/models/replica/av_nav/0301/classifier_reg"
file_path = os.path.join(path, "train.log")

keys = ["regressor_loss"]
value = {}
for key in keys:
    value[key] = []

freq = 1  # 用来平滑曲线，每freq个数据点取均值画一个图像点
threshold_episode = 0  # 从第threshold_episode开始统计

count = 0
sum = 0
threshold = threshold_episode * 4

with open(file_path) as f:
    lines = f.readlines()
    for line in lines:
        for key in keys:
            if key in line:
                pos = line.find(key)
                if line[pos - 1] == ' ':  # 有个softspl需要排除
                    count += 1
                    if count > threshold:
                        sum += float(line[pos + 2 + len(key):])
                        if (count % freq == 0):
                            value[key].append(sum / freq)
                            sum = 0

# x = []
# for i in range(len(value[keys[0]])):
#     x.append(i)

for key in keys:
    plt.hist(value[key], bins=30, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.legend()
plt.show()
plt.savefig(f'{path}/hist{keys[0]}.jpg', dpi=400, bbox_inches='tight')
plt.cla()
print("done")
