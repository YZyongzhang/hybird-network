import gzip
import json
import os
import numpy as np
import re
import shutil


sound_ids = np.load("data/sounds/sound_ids_1s_all_not_merge_train.npy", allow_pickle=True).item()
ori_sound_num = len(sound_ids.keys())
final_sound_nums = [150, 300]
for final_sound_num in final_sound_nums:
    source_sound_dir = os.path.join("data", "sounds", "1s_all")
    sound_dir = os.path.join("data", "sounds", "1s_all_%d" % final_sound_num)
    if os.path.isdir(sound_dir):
        shutil.rmtree(sound_dir)
    shutil.copytree(source_sound_dir, sound_dir)
    sound_num = ori_sound_num
    sound_dir_1 = os.path.join("sound_data", "juliantoquica", "sounds_wav")
    sound_dir_2 = os.path.join("sound_data", "Bitbeast", "sounds_wav")
    if sound_num < final_sound_num:
        for file in os.listdir(sound_dir_1):
            sound_name = file.split(".")[0]
            shutil.copy(os.path.join(sound_dir_1, file), sound_dir)
            sound_ids[sound_name] = sound_num
            sound_num += 1
            if sound_num >= final_sound_num:
                break
    if sound_num < final_sound_num:
        for file in os.listdir(sound_dir_2):
            sound_name = file.split(".")[0]
            shutil.copy(os.path.join(sound_dir_2, file), sound_dir)
            sound_ids[sound_name] = sound_num
            sound_num += 1
            if sound_num >= final_sound_num:
                break
    np.save("data/sounds/sound_ids_1s_all_not_merge_train_%d.npy" % final_sound_num, sound_ids)
