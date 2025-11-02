"""
REGISTER
"""
import torch
from habitat import Config
from ss_baselines.common.environments import AudioNavRLEnv
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from soundspaces import SoundSpacesSim , ContinuousSoundSpacesSim
from utils import draw_map
import pickle , copy , os ,random
import numpy as np
import habitat_sim
import math
class REGISTER:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"{name} has already in registry")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        
        if name not in self._registry:
            raise KeyError(f"{name} not in registry : {list(self._registry.keys())}")
        return self._registry[name]

    def create(self, name, *args, **kwargs):
       
        cls = self.get(name)
        return cls(*args, **kwargs)

    def available(self):
        return list(self._registry.keys())
    
CollectRegister = REGISTER()

@CollectRegister.register('angle')
class AngleCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            obs = self.env.reset()
            self.save(obs = obs)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None

@CollectRegister.register("greedy")
class GreedyCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            path_point = []
            obs = self.env.reset()
            action_id = self.sim.compute_oracle_actions()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs,action_id=action_id)
            path_point.append(self.sim.get_agent_state().position)
            for action in action_id:
                obs , reward , done , info = self.env.step(action=action)
                self.save(obs=obs,reward=reward,done=done,info=info)
                path_point.append(self.sim.get_agent_state().position)
                if action == action_id[-1]:
                    print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
            
            self.save(map=draw_map(self.env , path_point))
            self.save(path_point=path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None
        

@CollectRegister.register("collided")
class CollidedCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
        self.hybird_network = kwargs['model']
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            
            self.save_data = copy.deepcopy(self.save_data_struct)
            greedy_path_point = []
            collided_path_point = []
            obs = self.env.reset()
            pre_obs = obs
            rgb = torch.from_numpy(obs['rgb']).float() / 255.0
            depth = torch.from_numpy(obs['depth']).float()
            audio = torch.from_numpy(obs['spectrogram'][0]).float()
            action = self.hybird_network(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
            action = torch.argmax(action).item()
            done = False
            greedy_path_point.append(self.sim.get_agent_state().position)
            while not done:
                obs , rewad , done , info = self.env.step(action = action)
                pre_obs = obs
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                action = self.hybird_network(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
                action = torch.argmax(action).item()
                collided = self.sim.previous_step_collided
                if collided:
                    greedy_action = self.sim.compute_oracle_actions()
                    self.save(obs = obs)
                    for action in greedy_action[:3]:
                        obs , reward , done , info = self.env.step(action = action)
                        collided_path_point.append(self.sim.get_agent_state().position)
                        self.save(obs = obs , reward=reward,done=done,info=info,action_id=action)
                        
                        if done:
                            break
                    self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
                    break
            
            
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None

@CollectRegister.register("random")
class RandomCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            greedy_path_point = []
            collided_path_point = []
            obs = self.env.reset()
            
            done = False
            self.save(obs=obs)
            self.save(sound_id = self.env._env.current_episode.sound_id)
            greedy_path_point.append(self.sim.get_agent_state().position)
            while len(self.save_data['done']) < 50 and not done:
                action = random.choice([1,2,3])
                obs , reward , done , info = self.env.step(action = action)
                greedy_path_point.append(self.sim.get_agent_state().position)
                self.save(obs=obs,reward=reward,done=done,info=info,action_id=action)
                if done:
                    break
            
            self.save(greedy_map=draw_map(self.env , greedy_path_point))
            self.save(collided_map=draw_map(self.env , collided_path_point))
            self.save(greedy_path_point=greedy_path_point)
            self.save(collided_path_point=collided_path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
            
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineRL")
class OfflineCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
        self.hybird_network = kwargs['model']
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            # 40% 完全greedy 40% hybirdnetwork 20 % random
            epsilon = random.random()
            if epsilon < 0.4:
                # greedy
                self.save_data = copy.deepcopy(self.save_data_struct)
                path_point = []
                obs = self.env.reset()
                action_id = self.sim.compute_oracle_actions()
                done = False
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                self.save(obs=obs,action_id=action_id)
                path_point.append(self.sim.get_agent_state().position)
                for action in action_id:
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if action == action_id[-1]:
                        print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
                
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("greedy" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
                
            elif epsilon >= 0.4 and epsilon < 0.8:
                # hybird network 
                self.save_data = copy.deepcopy(self.save_data_struct)
                step = 0
                pre_action_collided = False
                path_point = []
                obs = self.env.reset()
                self.save(obs=obs)

                done = False
                
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                path_point.append(self.sim.get_agent_state().position)
                while not done:
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    with torch.no_grad():
                        logits = self.hybird_network(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
                    action = torch.argmax(logits).item()
                    self.save(action_id = action)

                    obs , reward , done , info = self.env.step(action=action)
                    if self.sim.previous_step_collided:
                        reward -= 1
                        pre_action_collided = self.sim.previous_step_collided
                    elif pre_action_collided:
                        pre_action_collided = False
                        reward +=1
                    step +=1
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done or step > 20:
                        print(f"action is {action} , done is {done} , info is {info}")
                        break
                if not done:
                    action_id = self.sim.compute_oracle_actions()
                    for action in action_id:
                        obs , reward , done , info = self.env.step(action=action)
                        self.save(action_id = action)
                        
                        self.save(obs=obs,reward=reward,done=done,info=info)
                        path_point.append(self.sim.get_agent_state().position)
                        if done:
                            # 这个地方会出现一次超过最大步数的done。所以如果没有break就会报错
                            print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
                            break
                
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("hybird" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
                
            elif epsilon >= 0.8 :
                self.save_data = copy.deepcopy(self.save_data_struct)
                path_point = []
                obs = self.env.reset()
                done = False
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                self.save(obs=obs)
                path_point.append(self.sim.get_agent_state().position)
                step = 0
                while step < 50:
                    action = random.choice([1,2,3])
                    self.save(action_id = action)
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done:
                        break
                    
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("random" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None
@CollectRegister.register("random_v2") # 这个不知道都是啥
class RandomCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            greedy_path_point = []
            collided_path_point = []
            obs = self.env.reset()
            
            done = False
            self.save(obs=obs)
            self.save(sound_id = self.env._env.current_episode.sound_id)
            greedy_path_point.append(self.sim.get_agent_state().position)
            while len(self.save_data['done']) < 50 and not done:
                action = random.choice([1,2,3])
                obs , reward , done , info = self.env.step(action = action)
                greedy_path_point.append(self.sim.get_agent_state().position)
                self.save(obs=obs,reward=reward,done=done,info=info,action_id=action)
                if done:
                    break
            
            self.save(greedy_map=draw_map(self.env , greedy_path_point))
            self.save(collided_map=draw_map(self.env , collided_path_point))
            self.save(greedy_path_point=greedy_path_point)
            self.save(collided_path_point=collided_path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
            
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineRLtwoframe")
class OfflineCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
        self.env = env
        self.sim: SoundSpacesSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
        self.hybird_network = kwargs['model']
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            # 40% 完全greedy 40% hybirdnetwork 20 % random
            epsilon = random.random()
            if epsilon < 0.4:
                # greedy
                self.save_data = copy.deepcopy(self.save_data_struct)
                path_point = []
                obs = self.env.reset()
                action_id = self.sim.compute_oracle_actions()
                done = False
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                self.save(obs=obs,action_id=action_id)
                path_point.append(self.sim.get_agent_state().position)
                for action in action_id:
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if action == action_id[-1]:
                        print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
                
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("greedy" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
                
            elif epsilon >= 0.4 and epsilon < 0.8:
                # hybird network 
                self.save_data = copy.deepcopy(self.save_data_struct)
                step = 0
                pre_action_collided = False
                path_point = []
                obs = self.env.reset()
                pre_rgb = torch.zeros_like(torch.from_numpy(obs['rgb']).float() / 255.0)
                pre_depth = torch.zeros_like(torch.from_numpy(obs['depth']).float())
                self.save(obs=obs)

                done = False
                
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                path_point.append(self.sim.get_agent_state().position)
                while not done:
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    trgb = torch.cat([pre_rgb , rgb] , dim =2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)

                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    with torch.no_grad():
                        logits = self.hybird_network(audio.to('cuda') , trgb.to('cuda') , tdepth.to('cuda'))
                    pre_rgb = rgb
                    pre_depth = depth
                    action = torch.argmax(logits).item()
                    self.save(action_id = action)
                    obs , reward , done , info = self.env.step(action=action)
                    if self.sim.previous_step_collided:
                        reward -= 1
                        pre_action_collided = self.sim.previous_step_collided
                    elif pre_action_collided:
                        pre_action_collided = False
                        reward +=1
                    step +=1
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done or step > 20:
                        print(f"action is {action} , done is {done} , info is {info}")
                        break
                if not done:
                    action_id = self.sim.compute_oracle_actions()
                    for action in action_id:
                        obs , reward , done , info = self.env.step(action=action)
                        self.save(action_id = action)
                        
                        self.save(obs=obs,reward=reward,done=done,info=info)
                        path_point.append(self.sim.get_agent_state().position)
                        if done:
                            # 这个地方会出现一次超过最大步数的done。所以如果没有break就会报错
                            print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
                            break
                
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("hybird" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
                
            elif epsilon >= 0.8 :
                self.save_data = copy.deepcopy(self.save_data_struct)
                path_point = []
                obs = self.env.reset()
                done = False
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                self.save(obs=obs)
                path_point.append(self.sim.get_agent_state().position)
                step = 0
                while step < 50:
                    action = random.choice([1,2,3])
                    self.save(action_id = action)
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done:
                        break
                    
                self.save(map=draw_map(self.env , path_point))
                self.save(path_point=path_point)
                self.store("random" , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineLevel")
class OffflineLevel:
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
            self.env = env
            self.sim: SoundSpacesSim = env._env._sim
            self.save_data_struct = config.DATA_STRUCT
            self.save_data_dir = config.DATA_DIR
            self.save_path_img = config.IMG_DIR
            self.save_path_type = config.IMG_TYPE
            self.level_rates = {
                'level1':(0  ,  1),
                'level2':(0.1 , 1),
                'level3':(0.2 , 2),
                'level4':(0.3 , 3),
                'level5':(0.4 , 4)
            }# (rate , step )
            self.level = ['level1' , 'level2' , 'level3' , 'level4' , 'level5']
            self.action_list = [1,2,3] # 去除step的0
    def collect(self): 
        for _ in range(self.env._env.number_of_episodes):
            # 40% 完全greedy 40% hybirdnetwork 20 % random

            level_name = random.choice(self.level)
            level_rate , level_step = self.level_rates[level_name]
            self.save_data = copy.deepcopy(self.save_data_struct)
            path_point = []
            obs = self.env.reset()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs)
            path_point.append(self.sim.get_agent_state().position)
            while not done:
                action_id = self.sim.compute_oracle_actions() # 获取greedy的action list
                self.save(greedy_action = action_id)
                for action_step , action in enumerate(action_id):
                    # 针对每一步进行level筛选
                    if action_step + 1 >= level_step: # 将index转化为step num ， 每一次等于level_step
                        # 进行概率判断
                        episilon = random.random() 
                        if episilon <= level_rate:
                            action = random.choice(self.action_list) # 进行 noise 干扰 , 否则action不变
                            self.save(action_id = action)
                            obs , reward , done , info = self.env.step(action=action)
                            self.save(obs=obs,reward=reward,done=done,info=info)
                            path_point.append(self.sim.get_agent_state().position)
                            break # 终止循环。
                        else:
                            self.save(action_id = action)
                            obs , reward , done , info = self.env.step(action=action)
                            self.save(obs=obs,reward=reward,done=done,info=info)
                            path_point.append(self.sim.get_agent_state().position)
                    else:
                        # 按照正常的逻辑action进行step
                        self.save(action_id = action)
                        obs , reward , done , info = self.env.step(action=action)
                        self.save(obs=obs,reward=reward,done=done,info=info)
                        path_point.append(self.sim.get_agent_state().position)

            
            self.save(map=draw_map(self.env , path_point))
            self.save(path_point=path_point)
            self.store(level_name , self.env._env.current_episode.scene_id , self.env._env.current_episode)
                
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineversion1")
class OfflineVersion1:
    """
    在固定的greedy步数之后进行概率noise , 如果执行了noise , 
    该版本首先进行转向，随后进行前进。如果碰撞，则结束noise过程。
    """
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
            self.env = env
            self.sim: SoundSpacesSim = env._env._sim
            self.save_data_struct = config.DATA_STRUCT
            self.save_data_dir = config.DATA_DIR
            self.save_path_img = config.IMG_DIR
            self.save_path_type = config.IMG_TYPE
            self.level_rates = {
                'level1':(0  ,  2),
                'level2':(0.1 , 2),
                'level3':(0.2 , 2),
                'level4':(0.3 , 2),
                'level5':(0.4 , 2)
            }# (rate , step )
            self.level = ['level1' , 'level2' , 'level3' , 'level4' , 'level5']
            self.action_list = [1,2,3] # 去除step的0
    def collect(self): 
        for _ in range(self.env._env.number_of_episodes):
            # 40% 完全greedy 40% hybirdnetwork 20 % random

            level_name = random.choice(self.level)
            level_rate , level_step = self.level_rates[level_name]
            self.save_data = copy.deepcopy(self.save_data_struct)
            path_point = []
            obs = self.env.reset()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs)
            path_point.append(self.sim.get_agent_state().position)
            while not done:
                action_id = self.sim.compute_oracle_actions() # 获取greedy的action list
                self.save(greedy_action = action_id)
                for action_step , action in enumerate(action_id):
                    # 针对每一步进行level筛选
                    if action_step + 1 >= level_step: # 将index转化为step num ， 每一次等于level_step
                        # 进行概率判断
                        episilon = random.random() 
                        if episilon <= level_rate:
                            # 核心在这里实现action 干扰。
                            # 1. 首先进行sample 左右action
                            print(done)
                            done = self.noise_action(path_point)
                            break # 终止循环。
                        else:
                            self.save(action_id = action)
                            obs , reward , done , info = self.env.step(action=action)
                            self.save(obs=obs,reward=reward,done=done,info=info)
                            path_point.append(self.sim.get_agent_state().position)
                    else:
                        # 按照正常的逻辑action进行step
                        self.save(action_id = action)
                        obs , reward , done , info = self.env.step(action=action)
                        self.save(obs=obs,reward=reward,done=done,info=info)
                        path_point.append(self.sim.get_agent_state().position)

            
            self.save(map=draw_map(self.env , path_point))
            self.save(path_point=path_point)
            self.store(level_name , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    
    def noise_action(self , path_point):
        frist_action = random.choice([2,3]) #1. 获取到左右的action
        self.save(action_id = frist_action)
        obs , reward , done , info = self.env.step(action = frist_action)
        self.save(obs=obs,reward=reward,done=done,info=info)
        path_point.append(self.sim.get_agent_state().position)
        while not self.sim.previous_step_collided and not done: # 2. 如果没有碰撞，则始终执行前进
            action = 1 
            self.save(action_id = action)
            obs , reward , done , info = self.env.step(action = action) # 3. 执行前进策略
            self.save(obs=obs,reward=reward,done=done,info=info)
            path_point.append(self.sim.get_agent_state().position)

        return done                

    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None

@CollectRegister.register("offlineversion2")
class OfflineVersion2:
    """
    在固定的greedy步数之后进行概率noise , 如果执行了noise , 
    该版本进行从左到右的广度遍历。即先执行左转。碰撞之后转回原朝向，再次进行前走，之后右转。
    如果某一个方向成功，重复进行策略。
    考虑到前进数据过多，这里广度的时候取出前进的遍历。只有左右.为了避免转圈，左右也需要进行random的采集
    """
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
            self.env = env
            self.sim: SoundSpacesSim = env._env._sim
            self.save_data_struct = config.DATA_STRUCT
            self.save_data_dir = config.DATA_DIR
            self.save_path_img = config.IMG_DIR
            self.save_path_type = config.IMG_TYPE
            self.level_rates = {
                'level1':(0  ,  2),
                'level2':(0.1 , 2),
                'level3':(0.2 , 2),
                'level4':(0.3 , 2),
                'level5':(0.4 , 2)
            }# (rate , step )
            self.level = ['level1' , 'level2' , 'level3' , 'level4' , 'level5']
            self.action_list = [1,2,3] # 去除step的0
    def collect(self): 
        for _ in range(self.env._env.number_of_episodes):
            # 40% 完全greedy 40% hybirdnetwork 20 % random

            level_name = random.choice(self.level)
            level_rate , level_step = self.level_rates[level_name]
            self.save_data = copy.deepcopy(self.save_data_struct)
            path_point = []
            obs = self.env.reset()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs)
            path_point.append(self.sim.get_agent_state().position)
            while not done:
                action_id = self.sim.compute_oracle_actions() # 获取greedy的action list
                self.save(greedy_action = action_id)
                for action_step , action in enumerate(action_id):
                    # 针对每一步进行level筛选
                    if action_step + 1 >= level_step: # 将index转化为step num ， 每一次等于level_step
                        # 进行概率判断
                        episilon = random.random() 
                        if episilon <= level_rate:
                            # 核心在这里实现action 干扰。
                            # 1. 首先进行sample 左右action
                            print(done)
                            done = self.noise_action(path_point)
                            break # 终止循环。
                        else:
                            self.save(action_id = action)
                            print(done)
                            obs , reward , done , info = self.env.step(action=action)
                            self.save(obs=obs,reward=reward,done=done,info=info)
                            path_point.append(self.sim.get_agent_state().position)
                    else:
                        # 按照正常的逻辑action进行step
                        self.save(action_id = action)
                        obs , reward , done , info = self.env.step(action=action)
                        self.save(obs=obs,reward=reward,done=done,info=info)
                        path_point.append(self.sim.get_agent_state().position)

            
            self.save(map=draw_map(self.env , path_point))
            self.save(path_point=path_point)
            self.store(level_name , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    
    def noise_action(self , path_point):
        takes_actions_num = 10 # 按照这种策略走10
        for i in range(takes_actions_num):
            frist_action = random.choice([2,3])
            r = self.take_one_noise_action(path_point , frist_action)
            if  r == 0:
                if frist_action == 3:
                    frist_action = 2
                else:
                    frist_action = 3
                    r = self.take_one_noise_action(path_point , frist_action)
                if r == 0: # 都没有成功。则打破本次noise
                    break
            elif r == 3:
                return True
    
    def take_one_noise_action(self , path_point , frist_action):

        self.save(action_id = frist_action)
        obs , reward , done , info = self.env.step(action = frist_action)
        if done:
            return 3
        self.save(obs=obs,reward=reward,done=done,info=info)
        path_point.append(self.sim.get_agent_state().position)
        next_action  = 1 # 2. 执行前进判断
        obs , reward , done , info = self.env.step(action = next_action) # 3. 尝试进行前进
        if done:
            return 3
        if self.sim.previous_step_collided: # 4. 如果前进碰撞了 ， 进行匹配逆转action
            if frist_action == 2 :
                frist_back_action = 3
            elif frist_action == 3:
                frist_back_action = 2
            self.env.step(action = frist_back_action) # 进行逆转
            return 0 # 没有成功执行
        else: # 这种情况下前进没有碰撞,保存数据
            self.save(obs=obs,reward=reward,done=done,info=info)
            path_point.append(self.sim.get_agent_state().position)
            return 1 # 成功执行


    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineversion3")
class OfflineVersion3:
    """
    version3 采取针对原声源位置进行递归回退找到mid sound point. 逐一进行导航
    """
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
            self.env = env
            self.sim: SoundSpacesSim = env._env._sim
            self.save_data_struct = config.DATA_STRUCT
            self.save_data_dir = config.DATA_DIR
            self.save_path_img = config.IMG_DIR
            self.save_path_type = config.IMG_TYPE
            self.level_rates = {
                'level1':(0  ,  2),
                'level2':(0.1 , 2),
                'level3':(0.2 , 2),
                'level4':(0.3 , 2),
                'level5':(0.4 , 2)
            }# (rate , step )
            self.level = ['level1' , 'level2' , 'level3' , 'level4' , 'level5']
            self.action_list = [1,2,3] # 去除step的0
            self.short_path_greedy = ShortestPathFollower(
                self.sim , 
                goal_radius=0, 
                return_one_hot=False , 
                stop_on_error=True
            ) # 这里的stop_on_error 会出现greedyerror. 如果为false的话。这个看看会有哪一种情况出现这种case
            self.short_path_greedy._build_follower()
    def mid_sound_point(self , total_sound , finnal_sound , distance ,geodesic_distance, nums):
        if nums == 0:
            return 1
        
        max_retry = 100
        mid_point = None

        for attempt in range(max_retry):
            candidate = self.sim.check_audio_point(self.sim.pathfinder.get_random_navigable_point_near(
                finnal_sound, distance, max_tries=100
            )).tolist()
            mid_geodesic_distance = self.sim.geodesic_distance(self.env._env.current_episode.start_position , [candidate])
            if math.isfinite(mid_geodesic_distance) and mid_geodesic_distance < geodesic_distance: # 不走回头路，确保geodesic大于midgeodesic
                mid_point = candidate
                break
            else:
                print(f"[Retry {attempt+1}/{max_retry}] Unreachable point {candidate}, resampling...")

        if mid_point is None:
            mid_point = finnal_sound # 如果都找不到中间点，则放弃。
        
        self.mid_sound_point(total_sound , mid_point , distance , geodesic_distance , nums - 1)
        total_sound.append(mid_point) # 保证geodesic较大的最后寻找
    

    def get_actions(self , goal_pos):

        # path = self.short_path_greedy._follower.find_path(goal_pos)
        self.sim.set_audio_point(goal_pos)
        actions = self.sim.compute_oracle_actions()

        return actions
    def remove_same_point(self , total_sound_point):
        seen = set()
        unique_points = []
        for p in total_sound_point:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique_points.append(p)
        total_sound_point = unique_points
        return total_sound_point
    
    def collect(self): 
        for _ in range(self.env._env.number_of_episodes):
            # 首先实现正常的版本，不分level
            # 1. 获取到原声源位置
            path_point = []
            self.save_data = copy.deepcopy(self.save_data_struct)
            finnal_sound_point = self.env._env.current_episode.goals[0].position # 这里进行debug看看属性。可以先跳过
            geodesic_distance = self.env._env.current_episode.info['geodesic_distance']
            nums_split = 3
            total_sound_point = []
            distance = geodesic_distance / nums_split
            self.mid_sound_point(total_sound_point , finnal_sound_point , distance , geodesic_distance , 3)
            total_sound_point.append(finnal_sound_point)
            total_sound_point = self.remove_same_point(total_sound_point)
            obs = self.env.reset()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs)
            self.save(total_sound_point = total_sound_point)
            for id , sound_point in enumerate(total_sound_point):
                actions = self.get_actions(sound_point)
                for action in actions:
                    if action == 0 and id != len(total_sound_point)-1:
                        # 如果action为0且不是最后一个声源点
                        break
                    self.save(action_id = action)
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done:
                        break
            self.save(map=draw_map(self.env , path_point , total_sound_point))
            self.save(path_point=path_point)
            self.store('level3' , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None


@CollectRegister.register("offlineversion_2.0")
class OfflineVersionSoundspaces2:
    """
    version3 采取针对原声源位置进行递归回退找到mid sound point. 逐一进行导航
    """
    def __init__(self , env:AudioNavRLEnv , config:Config , **kwargs):
            self.env = env
            self.sim: SoundSpacesSim = env._env._sim
            self.save_data_struct = config.DATA_STRUCT
            self.save_data_dir = config.DATA_DIR
            self.save_path_img = config.IMG_DIR
            self.save_path_type = config.IMG_TYPE
            self.level_rates = {
                'level1':(0  ,  2),
                'level2':(0.1 , 2),
                'level3':(0.2 , 2),
                'level4':(0.3 , 2),
                'level5':(0.4 , 2)
            }# (rate , step )
            self.level = ['level1' , 'level2' , 'level3' , 'level4' , 'level5']
            self.action_list = [1,2,3] # 去除step的0

    def reset_greedy_follower(self , ):
        self.greedy_follower = self.sim.make_greedy_follower(
            agent_id = 0,
            goal_radius = 1.0,
            stop_key = 0,
            forward_key = 1,
            left_key = 2,
            right_key = 3
            # stop_key="stop",
            # forward_key="move_forward",
            # left_key="turn_left",
            # right_key="turn_right",
        )
    def mid_sound_point(self , total_sound , finnal_sound , distance ,geodesic_distance, nums):
        if nums == 0:
            return 1
        
        max_retry = 100
        mid_point = None

        for attempt in range(max_retry):
            candidate = self.sim.pathfinder.get_random_navigable_point_near(
                finnal_sound, distance, max_tries=100
            ).tolist()            
            # candidate = self.sim.pathfinder.get_random_navigable_point().tolist()

            mid_geodesic_distance = self.sim.geodesic_distance(self.env._env.current_episode.start_position , [candidate])
            if math.isfinite(mid_geodesic_distance) and mid_geodesic_distance < geodesic_distance and self.sim.pathfinder.is_navigable(candidate): # 不走回头路，确保geodesic大于midgeodesic
                mid_point = candidate
                break
            else:
                print(f"[Retry {attempt+1}/{max_retry}] Unreachable point {candidate}, resampling...")

        if mid_point is None:
            mid_point = finnal_sound # 如果都找不到中间点，则放弃。
        
        # self.mid_sound_point(total_sound , mid_point , distance , geodesic_distance , nums - 1)
        total_sound.append(mid_point) # 保证geodesic较大的最后寻找
    

    def get_actions(self , goal_pos):
        # import pdb;pdb.set_trace()
        actions = self.greedy_follower.find_path(goal_pos)
        # actions , agent_pos , sound_pos = self.sim.pathfinder.find_path(goal_pos)

        return actions
    def remove_same_point(self , total_sound_point):
        seen = set()
        unique_points = []
        for p in total_sound_point:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique_points.append(p)
        total_sound_point = unique_points
        return total_sound_point
    
    def collect(self): 
        for _ in range(self.env._env.number_of_episodes):
            # 首先实现正常的版本，不分level
            # 1. 获取到原声源位置
            path_point = []
            self.save_data = copy.deepcopy(self.save_data_struct)
            # [-2.506288, -3.51, 1.086758]
            # finnal_sound_point = self.env._env.current_episode.goals[0].position # 这里进行debug看看属性。可以先跳过
            finnal_sound_point = self.env._env.current_episode.start_position
            geodesic_distance = self.env._env.current_episode.info['geodesic_distance']
            nums_split = 3
            total_sound_point = []
            # distance = geodesic_distance / nums_split
            distance = 10
            self.mid_sound_point(total_sound_point , finnal_sound_point , distance , geodesic_distance , 3)
            # total_sound_point.append(finnal_sound_point) # 会不会是finnnalpoint有时候sanmple不到地图上
            # total_sound_point = self.remove_same_point(total_sound_point)
            obs = self.env.reset()
            self.reset_greedy_follower()
            done = False
            self.save(sound_id = self.env._env.current_episode.info['sound'])
            self.save(obs=obs)
            self.save(total_sound_point = total_sound_point)
            for id , sound_point in enumerate(total_sound_point):
                # import pdb;pdb.set_trace()
                print(done)
                actions = self.get_actions(sound_point)
                for action in actions:
                    if action == 0 and id != len(total_sound_point)-1:
                        # 如果action为0且不是最后一个声源点
                        break
                    self.save(action_id = action)
                    obs , reward , done , info = self.env.step(action=action)
                    self.save(obs=obs,reward=reward,done=done,info=info)
                    path_point.append(self.sim.get_agent_state().position)
                    if done:
                        break
                if done:
                    break
            self.save(map=draw_map(self.env , path_point , total_sound_point))
            self.save(path_point=path_point)
            self.store('level3' , self.env._env.current_episode.scene_id , self.env._env.current_episode)
    
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, level , scene , id):
        os.makedirs(f"{self.save_data_dir}/{level}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{level}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None