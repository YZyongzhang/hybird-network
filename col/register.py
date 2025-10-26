"""
REGISTER
"""
import torch
from habitat import Config
from ss_baselines.common.environments import AudioNavRLEnv
from soundspaces import SoundSpacesSim
from utils import draw_map
import pickle , copy , os ,random
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
<<<<<<< HEAD
                path_point = []
                obs = self.env.reset()
                done = False
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                logits = self.hybird_network(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
                action = torch.argmax(logits).item()
                self.save(action_id = action)
                self.save(sound_id = self.env._env.current_episode.info['sound'])
                self.save(obs=obs)
                path_point.append(self.sim.get_agent_state().position)
                while not done:
                
                    obs , reward , done , info = self.env.step(action=action)
                    step +=1
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    logits = self.hybird_network(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
                    action = torch.argmax(logits).item()
                    self.save(action_id = action)
=======
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
>>>>>>> 8cdf5fcf58b8de4e1474607f603a41f22700168b
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
<<<<<<< HEAD
=======
                            break
>>>>>>> 8cdf5fcf58b8de4e1474607f603a41f22700168b
                
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