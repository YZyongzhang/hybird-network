import numpy as np
import os
import re

sound_ids = {}
real_sound_name_ids = {}
for root, dirs, files in os.walk("data/sounds/1s_all"):
    idx = 0
    for file in files:
        sound_name = file.split('.')[0]
        real_sound_name = sound_name #re.sub("^c_|[_0-9]*$", "", sound_name)
        if real_sound_name in real_sound_name_ids.keys():
            sound_ids[sound_name] = real_sound_name_ids[real_sound_name]
        else:
            sound_ids[sound_name] = idx
            real_sound_name_ids[real_sound_name] = idx
            idx += 1

np.save("data/sounds/sound_ids_1s_all_not_merge.npy", sound_ids)
