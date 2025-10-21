import copy
import torch
from tqdm import tqdm
import pickle
import os
from PIL import Image
class OnlineTest:
    def __init__(self , env , hybirdmodel):
        self.env = env
        self.sim = env._env._sim
        self.hybirdmodel = hybirdmodel
        
    def rollout(self , sac_model , logger):
        total_reward = 0
        spl = 0
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            path = f"{scene}_{episode_id}"
            os.makedirs(path , exist_ok=True)
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            with torch.no_grad():
                Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                visual = torch.from_numpy(obs['rgb']).float() / 255.0
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                state = self.hybirdmodel.embedding_forward(audio.to('cuda') , visual.to('cuda'))
                action_hybird = self.hybirdmodel(audio.to("cuda") , visual.to('cuda'))
                action_logits = sac_model(state.to('cuda'))
                action_sac = action_logits.argmax(dim=1).item()
                action_hybird = action_hybird.argmax(dim=1).item()
            while not done or step < 100:
                obs , reward , done , info = self.env.step(action=action_hybird)
                with torch.no_grad():
                    Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                    visual = torch.from_numpy(obs['rgb']).float() / 255.0
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    state = self.hybirdmodel.embedding_forward(audio.to("cuda") , visual.to('cuda'))
                    action_hybird = self.hybirdmodel(audio.to("cuda") , visual.to('cuda'))
                    action_sac = sac_model(state.to('cuda'))
                    action_sac = action_sac.argmax(dim=1).item()
                    action_hybird = action_hybird.argmax(dim=1).item()
                    
                logger.info(f"take action sac model {action_sac} ,take action hybird model {action_hybird}  reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                step +=1
                epsiode_reward +=reward
                if done or step >= 10:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes