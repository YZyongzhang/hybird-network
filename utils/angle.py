import numpy as np
import quaternion  # 需要安装 numpy-quaternion 库
def get_turn_angle(agent_pos, agent_rotation, source_pos):
    agent_pos = np.array(agent_pos, dtype=float)
    source_pos = np.array(source_pos, dtype=float)

    # 转成 quaternion 对象 (w,x,y,z)
    if isinstance(agent_rotation, (list, np.ndarray)):
        agent_rotation = np.quaternion(agent_rotation[0],
                                       agent_rotation[1],
                                       agent_rotation[2],
                                       agent_rotation[3])

    # 声音方向向量
    dir_to_source = source_pos - agent_pos
    
    dir_to_source[1] = 0  
    
    # 如果两点重合，直接返回 0
    if np.allclose(dir_to_source, 0):
        return 0.0

    dir_to_source = dir_to_source / np.linalg.norm(dir_to_source)

    # agent 朝向向量
    forward = quaternion.as_rotation_matrix(agent_rotation) @ np.array([0, 0, -1])
    forward[1] = 0
    forward = forward / np.linalg.norm(forward)
    print(forward)
    # 计算夹角
    dot = np.clip(np.dot(forward, dir_to_source), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))

    # 用叉积判断左右
    cross = np.cross(forward, dir_to_source)
    if cross[1] < 0:  # y轴负说明在右边
        angle = -angle

    return angle

def turn_angle(pose, sound_pos):
    # obs: [x, z, heading, t]
    agent_x, agent_z, heading = pose[0], pose[1], pose[2]
    sx , sy , sz = sound_pos

    # 声音方向向量
    dir_to_sound = np.array([sx - agent_x, sz - agent_z])
    if np.allclose(dir_to_sound, 0):
        return 0.0  # 声源就在当前位置

    dir_to_sound = dir_to_sound / np.linalg.norm(dir_to_sound)

    # agent 朝向向量
    forward = np.array([np.cos(heading), np.sin(heading)])

    # 夹角
    dot = np.clip(np.dot(forward, dir_to_sound), -1.0, 1.0)
    angle = np.arccos(dot)

    # 判断左右
    cross = forward[0]*dir_to_sound[1] - forward[1]*dir_to_sound[0]
    if cross < 0:
        angle = -angle

    return np.degrees(angle)

import matplotlib.pyplot as plt
import cv2

def draw_turn_on_image(image, agent_pos, heading, sound_pos, scale=50):
    """
    在输入的图片上绘制 agent、朝向、声源以及转角箭头
    
    Args:
        image: 背景图 (numpy array, HxWx3)
        agent_pos: (x, z)，agent位置
        heading: float，朝向角 (弧度)
        sound_pos: (sx, sz)，声源位置
        scale: 绘制箭头时的缩放倍数
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # 转换到图像坐标系 (假设原点在图像中心，x→右，z→下)
    def world_to_img(pt):
        x, z = pt
        u = int(w//2 + x * scale)
        v = int(h//2 - z * scale)  # z轴向上，所以取反
        return (u, v)
    
    agent_xy = world_to_img(agent_pos)
    sound_xy = world_to_img(sound_pos)
    
    # agent 朝向向量
    forward = np.array([np.cos(heading), np.sin(heading)])
    forward_end = world_to_img(agent_pos + forward*0.5)  # 0.5长度箭头
    
    # dir_to_sound 向量
    dir_to_sound = np.array(sound_pos) - np.array(agent_pos)
    if np.linalg.norm(dir_to_sound) > 1e-6:
        dir_to_sound /= np.linalg.norm(dir_to_sound)
    sound_dir_end = world_to_img(agent_pos + dir_to_sound*0.5)
    
    # 绘制 agent
    cv2.circle(img, agent_xy, 6, (255,0,0), -1)  # 蓝色点
    cv2.putText(img, "Agent", (agent_xy[0]+10, agent_xy[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
    # 绘制声源
    cv2.circle(img, sound_xy, 6, (0,0,255), -1)  # 红色点
    cv2.putText(img, "Sound", (sound_xy[0]+10, sound_xy[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    # 绘制朝向箭头
    cv2.arrowedLine(img, agent_xy, forward_end, (255,0,0), 2, tipLength=0.3)
    
    # 绘制指向声源箭头
    cv2.arrowedLine(img, agent_xy, sound_dir_end, (0,0,255), 2, tipLength=0.3)
    
    return img

