import copy
import torch
from tqdm import tqdm
class OnlineTest:
    def __init__(self , env , hybirdmodel):
        self.env = env
        self.hybirdmodel = hybirdmodel
        
    def rollout(self , sac_model , logger):
        total_reward = 0
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            visual = torch.from_numpy(obs['rgb']).float() / 255.0
            audio = torch.from_numpy(obs['spectrogram'][0]).float()
            state = self.hybirdmodel.embedding_forward(audio.to('cuda') , visual.to('cuda'))
            action_logits = sac_model(state.to('cuda'))
            action = action_logits.argmax(dim=1).item()
            while not done or step < 100:
                obs , reward , done , info = self.env.step(action=action)
                visual = torch.from_numpy(obs['rgb']).float() / 255.0
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                state = self.hybirdmodel.embedding_forward(audio.to("cuda") , visual.to('cuda'))
                action_logits = sac_model(state.to('cuda'))
                action = action_logits.argmax(dim=1).item()
                logger.info(f"take action {action} , reward {reward} , step {step} , done {done}")
                step +=1
                epsiode_reward +=reward
                if done or step >= 100:
                    logger.info(f"episode is done , info is {info} \nsumreward is {epsiode_reward}")
                    break
            total_reward +=  epsiode_reward
        return total_reward/self.env._env.number_of_episodes