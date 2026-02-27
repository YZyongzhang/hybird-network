import copy
import torch
from utils.visualizations import draw_sound , plot_top_down_map , draw_point
from habitat.utils.visualizations import maps
from tqdm import tqdm
import pickle
import os
from PIL import Image
from collections import deque
class OnlineTest:
    def __init__(self , env , hybirdmodel , config ):
        self.env = env
        self.sim = env._env._sim
        self.hybirdmodel = hybirdmodel
        self.config = config
        
    def rollout_1(self ,epoch ,  sac_model , logger):
        total_reward = 0
        spl = 0
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            path = f"img/{scene}_{episode_id}"
            os.makedirs(path , exist_ok=True)
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            with torch.no_grad():
                Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                # state = self.hybirdmodel.embedding_forward(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                # action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                state = self.hybirdmodel.embedding_forward(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda'))
                action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda'))
                action_sac = sac_model.get_action(state.to('cuda'))
                action_hybird = action_hybird.argmax(dim=1).item()
                # pre_rgb = rgb
                # pre_depth = depth
            while not done or step < 200:
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac}, take action hybird model {action_hybird} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    # state = self.hybirdmodel.embedding_forward(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') ,  pre_rgb.to('cuda') , pre_depth.to('cuda'))
                    # action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                    state = self.hybirdmodel.embedding_forward(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda'))
                    action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda'))
                    action_sac = sac_model.get_action(state.to('cuda'))
                    action_hybird = action_hybird.argmax(dim=1).item()
                    # pre_rgb = rgb
                    # pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= 100:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes
    def rollout_two_frame(self , epoch , sac_model , logger):
        total_reward = 0
        spl = 0
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            path = f"img/{scene}_{episode_id}"
            os.makedirs(path , exist_ok=True)
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            with torch.no_grad():
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                state = self.hybirdmodel.embedding_forward(audio.to('cuda') , trgb.to('cuda') , tdepth.to('cuda'))
                action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                action_sac = sac_model.get_action(state.to('cuda'))
                action_hybird = action_hybird.argmax(dim=1).item()
                pre_rgb = rgb
                pre_depth = depth
            while not done or step < 200:
                # obs , reward , done , info = self.env.step(action=action_sac)
                obs , reward , done , info = self.env.step(action=action_hybird)
                logger.info(f"take action sac model {action_sac}, take action hybird model {action_hybird} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    state = self.hybirdmodel.embedding_forward(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    action_sac = sac_model.get_action(state.to('cuda'))
                    action_hybird = action_hybird.argmax(dim=1).item()
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= 200:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    top_down_map = plot_top_down_map(info)
                    draw_point(self.sim , self.env._env.current_episode.start_position ,  top_down_map)
                    draw_point(self.sim , self.env._env.current_episode.goals[0].position ,  top_down_map)
                    os.makedirs(f"./spl/{epoch}" , exist_ok=True)
                    Image.fromarray(top_down_map).save(f"./spl/{epoch}/{self.env._env.current_episode.scene_id[-15:-4]}_{self.env._env.current_episode.episode_id}.png")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes
    def rollout_hybrid_offline(self , epoch , sac_model , logger):
        total_reward = 0.0
        total_spl = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_steps = int(getattr(self.config, "MAX_STEPS", 200))
        num_eps = int(self.env._env.number_of_episodes)

        def _to_env_action(a):
            if isinstance(a, torch.Tensor):
                if a.numel() == 1:
                    return int(a.item())
                return int(a.reshape(-1)[0].item())
            return int(a)

        for _ in tqdm(range(num_eps), desc="onlinetest"):
            obs = self.env.reset()
            current_ep = self.env._env.current_episode
            scene = current_ep.scene_id[-15:-4]
            episode_id = current_ep.episode_id

            logger.info(f"scene is {scene}, episodeid is {episode_id}")
            episode_reward = 0.0
            done = False
            step = 0
            info = {"spl": 0.0, "distance_to_goal": -1.0}

            with torch.no_grad():
                rgb = torch.from_numpy(obs["rgb"]).float() / 255.0
                depth = torch.from_numpy(obs["depth"]).float()
                audio = torch.from_numpy(obs["spectrogram"][0]).float()
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb, rgb], dim=2)
                tdepth = torch.cat([pre_depth, depth], dim=2)
                action_sac = _to_env_action(
                    sac_model.get_action((audio.to(device), trgb.to(device), tdepth.to(device)))
                )
                pre_rgb = rgb
                pre_depth = depth

            while (not done) and (step < max_steps):
                obs, reward, done, info = self.env.step(action=action_sac)
                logger.info(
                    f"take action sac model {action_sac}, reward {reward}, step {step}, "
                    f"done {done}, is collided {self.sim.previous_step_collided}"
                )

                episode_reward += float(reward)
                step += 1
                if done or step >= max_steps:
                    break

                with torch.no_grad():
                    rgb = torch.from_numpy(obs["rgb"]).float() / 255.0
                    depth = torch.from_numpy(obs["depth"]).float()
                    audio = torch.from_numpy(obs["spectrogram"][0]).float()
                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    action_sac = _to_env_action(
                        sac_model.get_action((audio.to(device), trgb.to(device), tdepth.to(device)))
                    )
                    pre_rgb = rgb
                    pre_depth = depth

            logger.info(
                f"episode is done, distance_to_goal is {info.get('distance_to_goal', -1)}, "
                f"spl is {info.get('spl', 0.0)} \nsumreward is {episode_reward}"
            )

            total_reward += episode_reward
            total_spl += float(info.get("spl", 0.0))

        return total_reward / num_eps, total_spl / num_eps
    def rollout_lstm(self , epoch , sac_model , logger):
        """
        twoframe + lstm
        """

        total_reward = 0
        spl = 0
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            path = f"img/{scene}_{episode_id}"
            os.makedirs(path , exist_ok=True)
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            ht = None
            ct = None
            with torch.no_grad():
                Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                # 获取hybrid model的输出
                state = self.hybirdmodel.embedding_forward(audio.to('cuda') , trgb.to('cuda') , tdepth.to('cuda'))
                action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                action_hybird = action_hybird.argmax(dim=1).item()
                # 获取offlineRL 的输出

                state, ht, ct = sac_model.hybrid_LSTM.inference(state.unsqueeze(0), ht, ct) # state batch ， time ， 256
                action_sac = sac_model.get_action(state.to('cuda'), eval=True).item()
                
                pre_rgb = rgb
                pre_depth = depth
            while not done or step < 100:
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac}, take action hybird model {action_hybird} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    state = self.hybirdmodel.embedding_forward(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    action_hybird = action_hybird.argmax(dim=1).item()

                    state, ht, ct = sac_model.hybrid_LSTM.inference(state.unsqueeze(0), ht, ct)
                    action_sac = sac_model.get_action(state.to('cuda') , eval=True).item()
                    
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= 100:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes
    def rollout_lstm_attention(self , epoch , sac_model , logger):
        """
        twoframe + lstm
        """

        total_reward = 0
        spl = 0
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            obs = self.env.reset()

            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            path = f"img/{scene}_{episode_id}"
            os.makedirs(path , exist_ok=True)
            
            logger.info(f"{epoch} , scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            done = False
            step = 0
            ht_audio = None
            ct_audio = None
            ht_visual_audio = None
            ct_visual_audio = None
            with torch.no_grad():
                Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = torch.from_numpy(obs['spectrogram'][0]).float()
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                # 获取hybrid model的输出
                audio_encoder , visual_audio_encoder = self.hybirdmodel.embedding_forward_attention(audio.to('cuda') , trgb.to('cuda') , tdepth.to('cuda'))
                action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                action_hybird = action_hybird.argmax(dim=1).item()
                # 获取offlineRL 的输出
                visual_audio_encoder = visual_audio_encoder.unsqueeze(0)
                # import pdb;pdb.set_trace()
                audio_state, ht_audio, ct_audio = sac_model.hybrid_LSTM.inference(audio_encoder.unsqueeze(0), ht_audio, ct_audio) # state batch ， time ， 256
                visual_audio_state, ht_visual_audio, ct_visual_audio = sac_model.hybrid_LSTM.inference(visual_audio_encoder.unsqueeze(0), ht_visual_audio, ct_visual_audio) # state batch ， time ， 256
                state = sac_model.attention(audio_state , visual_audio_state)

                action_sac = sac_model.get_action(state.to('cuda'), eval=True).item()
                
                pre_rgb = rgb
                pre_depth = depth
            while not done or step < 100:
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac}, take action hybird model {action_hybird} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    Image.fromarray(obs['rgb']).save(f"{path}/{step}.png")
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = torch.from_numpy(obs['spectrogram'][0]).float()
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    audio_encoder , visual_audio_encoder = self.hybirdmodel.embedding_forward_attention(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    visual_audio_encoder = visual_audio_encoder.unsqueeze(0)
                    action_hybird = self.hybirdmodel(audio.to("cuda") , trgb.to('cuda') , tdepth.to('cuda'))
                    action_hybird = action_hybird.argmax(dim=1).item()

                    audio_state, ht_audio, ct_audio = sac_model.hybrid_LSTM.inference(audio_encoder.unsqueeze(0), ht_audio, ct_audio) # state batch ， time ， 256
                    visual_audio_state, ht_visual_audio, ct_visual_audio = sac_model.hybrid_LSTM.inference(visual_audio_encoder.unsqueeze(0), ht_visual_audio, ct_visual_audio) # state batch ， time ， 256
                    state = sac_model.attention(audio_state , visual_audio_state)
                    action_sac = sac_model.get_action(state.to('cuda') , eval=True).item()
                    
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= 100:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    # top_down_map = plot_top_down_map(info)
                    # draw_point(self.sim , self.env._env.current_episode.start_position ,  top_down_map)
                    # draw_point(self.sim , self.env._env.current_episode.goals[0].position ,  top_down_map)
                    # os.makedirs(f"./online_img/{epoch}" , exist_ok=True)
                    # Image.fromarray(top_down_map).save(f"./online_img/{epoch}/{self.env._env.current_episode.scene_id[-15:-4]}_{self.env._env.current_episode.episode_id}.png")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes

    def rollout_lstm_v15(self, epoch, sac_model, logger):
        """
        v1_5: hybrid embedding (two-frame) + offline LSTM policy over encoded sequence.
        """
        total_reward = 0.0
        total_spl = 0.0
        seq_len = int(getattr(self.config, "lstm_seq_len", 5))
        max_steps = int(getattr(self.config, "MAX_STEPS", 200))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for _ in tqdm(range(self.env._env.number_of_episodes), desc="onlinetest"):
            obs = self.env.reset()
            scene = self.env._env.current_episode.scene_id[-15:-4]
            episode_id = self.env._env.current_episode.episode_id
            logger.info(f"{epoch} , scene is {scene}  , episodeid is {episode_id}")

            done = False
            step = 0
            epsiode_reward = 0.0
            info = {"spl": 0.0, "distance_to_goal": -1.0}
            pre_rgb = None
            pre_depth = None
            state_queue = deque(maxlen=seq_len)

            while (not done) and (step < max_steps):
                with torch.no_grad():
                    rgb = torch.from_numpy(obs["rgb"]).float() / 255.0
                    depth = torch.from_numpy(obs["depth"]).float()
                    audio = torch.from_numpy(obs["spectrogram"][0]).float()
                    if pre_rgb is None:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    state = self.hybirdmodel.embedding_forward(
                        audio.to(device), trgb.to(device), tdepth.to(device)
                    ).detach().cpu()
                    state_queue.append(state)

                    while len(state_queue) < seq_len:
                        state_queue.appendleft(torch.zeros_like(state))
                    seq_state = torch.stack(list(state_queue), dim=0).unsqueeze(0)
                    action_sac = int(sac_model.get_action(seq_state.to(device), eval=True))
                    action_hybird = int(
                        self.hybirdmodel(audio.to(device), trgb.to(device), tdepth.to(device))
                        .argmax(dim=1)
                        .item()
                    )
                    pre_rgb = rgb
                    pre_depth = depth

                obs, reward, done, info = self.env.step(action=action_sac)
                logger.info(
                    f"take action sac model {action_sac}, take action hybird model {action_hybird} ,"
                    f"reward {reward} , step {step} , done {done} , "
                    f"is collided {self.sim.previous_step_collided}"
                )
                epsiode_reward += float(reward)
                step += 1

            logger.info(
                f"episode is done , distance_to_goal is {info['distance_to_goal']}, "
                f"spl is {info['spl']} \nsumreward is {epsiode_reward}"
            )
            total_reward += epsiode_reward
            total_spl += float(info.get("spl", 0.0))

        eps = float(self.env._env.number_of_episodes)
        return total_reward / eps, total_spl / eps
    

    def rollout(self , epoch , sac_model , logger):
        if self.config.model == "v1":
            return self.rollout_two_frame(epoch , sac_model , logger)
        elif self.config.model == "v1_3":
            return self.rollout_hybrid_offline(epoch , sac_model , logger)
        elif self.config.model == 'v2':
            return self.rollout_two_frame(epoch , sac_model , logger)
        elif self.config.model == 'v4':
            return self.rollout_lstm(epoch , sac_model , logger)
        elif self.config.model == 'v5':
            return self.rollout_lstm_attention(epoch , sac_model , logger)
        elif self.config.model == 'v1_5':
            return self.rollout_lstm_v15(epoch, sac_model, logger)
    
