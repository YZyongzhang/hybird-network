import pickle , os
# path = '../dataset/pickle/v2/greedy(hybirdnetwork_with_RGBD_collided)/hybird/5q7pvUzZiYa/976.pkl'
path = './dataset/pickle/v2/offline(level)/level5/PX4nDJXEHrG/6256.pkl'
with open(path , 'rb') as f:
    data = pickle.load(f)
import sys
sys.path.append("/home/kongxiangyu/yongzhang/finnal_exp")
from utils.visualizations import plot_top_down_map
from PIL import Image
Image.fromarray(data['map'][0]).save('./a.png')
import pdb;pdb.set_trace()