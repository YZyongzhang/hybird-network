import torch
from tqdm import tqdm
from collections import deque


class OnlineTest:
    def __init__(self , env , hybirdmodel , config ):
        self.env = env
        self.sim = env._env._sim
        self.hybirdmodel = hybirdmodel
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_env_action(self, action):
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                return int(action.item())
            return int(action.reshape(-1)[0].item())
        return int(action)

    def _get_audio_tensor(self, obs):
        spec = obs["spectrogram"]
        if isinstance(spec, (tuple, list)):
            spec = spec[0]
        return torch.from_numpy(spec).float()
        
    def rollout_1(self ,epoch ,  sac_model , logger):
        total_reward = 0
        spl = 0
        max_steps = int(getattr(self.config, "MAX_STEPS", 100))
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            with torch.no_grad():
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = self._get_audio_tensor(obs)
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                # state = self.hybirdmodel.embedding_forward(audio.to('cuda') , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                # action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                state = self.hybirdmodel.embedding_forward(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device))
                action_hybird = self.hybirdmodel(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device))
                action_sac = self._to_env_action(sac_model.get_action(state.to(self.device)))
                action_hybird = action_hybird.argmax(dim=1).item()
                # pre_rgb = rgb
                # pre_depth = depth
            while (not done) and (step < max_steps):
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = self._get_audio_tensor(obs)
                    # state = self.hybirdmodel.embedding_forward(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') ,  pre_rgb.to('cuda') , pre_depth.to('cuda'))
                    # action_hybird = self.hybirdmodel(audio.to("cuda") , rgb.to('cuda') , depth.to('cuda') , pre_rgb.to('cuda') , pre_depth.to('cuda'))
                    state = self.hybirdmodel.embedding_forward(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device))
                    action_hybird = self.hybirdmodel(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device))
                    action_sac = self._to_env_action(sac_model.get_action(state.to(self.device)))
                    action_hybird = action_hybird.argmax(dim=1).item()
                    # pre_rgb = rgb
                    # pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= max_steps:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes
    def rollout_two_frame(self , epoch , sac_model , logger):
        total_reward = 0
        spl = 0
        max_steps = int(getattr(self.config, "MAX_STEPS", 200))
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            with torch.no_grad():
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = self._get_audio_tensor(obs)
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                state = self.hybirdmodel.embedding_forward(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_sac = self._to_env_action(sac_model.get_action(state.to(self.device)))
                action_hybird = action_hybird.argmax(dim=1).item()
                pre_rgb = rgb
                pre_depth = depth
            while (not done) and (step < max_steps):
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = self._get_audio_tensor(obs)
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    state = self.hybirdmodel.embedding_forward(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    action_sac = self._to_env_action(sac_model.get_action(state.to(self.device)))
                    action_hybird = action_hybird.argmax(dim=1).item()
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= max_steps:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
                    break
            total_reward +=  epsiode_reward
            spl += info['spl']
            
        return total_reward/self.env._env.number_of_episodes , spl / self.env._env.number_of_episodes
    def rollout_hybrid_offline(self , epoch , sac_model , logger):
        total_reward = 0.0
        total_spl = 0.0
        max_steps = int(getattr(self.config, "MAX_STEPS", 200))
        num_eps = int(self.env._env.number_of_episodes)

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
                audio = self._get_audio_tensor(obs)
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb, rgb], dim=2)
                tdepth = torch.cat([pre_depth, depth], dim=2)
                action_sac = self._to_env_action(
                    sac_model.get_action((audio.to(self.device), trgb.to(self.device), tdepth.to(self.device)))
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
                    audio = self._get_audio_tensor(obs)
                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    action_sac = self._to_env_action(
                        sac_model.get_action((audio.to(self.device), trgb.to(self.device), tdepth.to(self.device)))
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
        max_steps = int(getattr(self.config, "MAX_STEPS", 100))
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            
            logger.info(f"scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            obs = self.env.reset()
            done = False
            step = 0
            ht = None
            ct = None
            with torch.no_grad():
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = self._get_audio_tensor(obs)
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                # 获取hybrid model的输出
                state = self.hybirdmodel.embedding_forward(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_hybird = action_hybird.argmax(dim=1).item()
                # 获取offlineRL 的输出

                state, ht, ct = sac_model.hybrid_LSTM.inference(state.unsqueeze(0), ht, ct) # state batch ， time ， 256
                action_sac = self._to_env_action(sac_model.get_action(state.to(self.device), eval=True))
                
                pre_rgb = rgb
                pre_depth = depth
            while (not done) and (step < max_steps):
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = self._get_audio_tensor(obs)
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    state = self.hybirdmodel.embedding_forward(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    action_hybird = action_hybird.argmax(dim=1).item()

                    state, ht, ct = sac_model.hybrid_LSTM.inference(state.unsqueeze(0), ht, ct)
                    action_sac = self._to_env_action(sac_model.get_action(state.to(self.device), eval=True))
                    
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= max_steps:
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
        max_steps = int(getattr(self.config, "MAX_STEPS", 100))
        
        for _ in tqdm(range(self.env._env.number_of_episodes),desc="onlinetest"):
            obs = self.env.reset()

            scene = self.env._env.current_episode.scene_id[-15:-4] 
            episode_id = self.env._env.current_episode.episode_id
            
            logger.info(f"{epoch} , scene is {scene}  , episodeid is {episode_id}")
            epsiode_reward = 0            
            done = False
            step = 0
            ht_audio = None
            ct_audio = None
            ht_visual_audio = None
            ct_visual_audio = None
            with torch.no_grad():
                rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                depth = torch.from_numpy(obs['depth']).float()
                audio = self._get_audio_tensor(obs)
                pre_rgb = torch.zeros_like(rgb)
                pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb , rgb] , dim=2)
                tdepth = torch.cat([pre_depth , depth] , dim=2)
                # 获取hybrid model的输出
                audio_encoder , visual_audio_encoder = self.hybirdmodel.embedding_forward_attention(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                action_hybird = action_hybird.argmax(dim=1).item()
                # 获取offlineRL 的输出
                visual_audio_encoder = visual_audio_encoder.unsqueeze(0)
                # import pdb;pdb.set_trace()
                audio_state, ht_audio, ct_audio = sac_model.hybrid_LSTM.inference(audio_encoder.unsqueeze(0), ht_audio, ct_audio) # state batch ， time ， 256
                visual_audio_state, ht_visual_audio, ct_visual_audio = sac_model.hybrid_LSTM.inference(visual_audio_encoder.unsqueeze(0), ht_visual_audio, ct_visual_audio) # state batch ， time ， 256
                state = sac_model.attention(audio_state , visual_audio_state)

                action_sac = self._to_env_action(sac_model.get_action(state.to(self.device), eval=True))
                
                pre_rgb = rgb
                pre_depth = depth
            while (not done) and (step < max_steps):
                obs , reward , done , info = self.env.step(action=action_sac)
                logger.info(f"take action sac model {action_sac} ,reward {reward} , step {step} , done {done} , is collided {self.sim.previous_step_collided}")
                
                with torch.no_grad():
                    rgb = torch.from_numpy(obs['rgb']).float() / 255.0
                    depth = torch.from_numpy(obs['depth']).float()
                    audio = self._get_audio_tensor(obs)
                    trgb = torch.cat([pre_rgb , rgb] , dim=2)
                    tdepth = torch.cat([pre_depth , depth] , dim=2)
                    audio_encoder , visual_audio_encoder = self.hybirdmodel.embedding_forward_attention(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    visual_audio_encoder = visual_audio_encoder.unsqueeze(0)
                    action_hybird = self.hybirdmodel(audio.to(self.device) , trgb.to(self.device) , tdepth.to(self.device))
                    action_hybird = action_hybird.argmax(dim=1).item()

                    audio_state, ht_audio, ct_audio = sac_model.hybrid_LSTM.inference(audio_encoder.unsqueeze(0), ht_audio, ct_audio) # state batch ， time ， 256
                    visual_audio_state, ht_visual_audio, ct_visual_audio = sac_model.hybrid_LSTM.inference(visual_audio_encoder.unsqueeze(0), ht_visual_audio, ct_visual_audio) # state batch ， time ， 256
                    state = sac_model.attention(audio_state , visual_audio_state)
                    action_sac = self._to_env_action(sac_model.get_action(state.to(self.device), eval=True))
                    
                    pre_rgb = rgb
                    pre_depth = depth
                step +=1
                epsiode_reward +=reward
                if done or step >= max_steps:
                    logger.info(f"episode is done , distance_to_goal is {info['distance_to_goal']}, spl is {info['spl']} \nsumreward is {epsiode_reward}")
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
                    audio = self._get_audio_tensor(obs)
                    if pre_rgb is None:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    state = self.hybirdmodel.embedding_forward(
                        audio.to(self.device), trgb.to(self.device), tdepth.to(self.device)
                    ).detach().cpu()
                    state_queue.append(state)

                    while len(state_queue) < seq_len:
                        state_queue.appendleft(torch.zeros_like(state))
                    seq_state = torch.stack(list(state_queue), dim=0).unsqueeze(0)
                    action_sac = self._to_env_action(sac_model.get_action(seq_state.to(self.device), eval=True))
                    action_hybird = int(
                        self.hybirdmodel(audio.to(self.device), trgb.to(self.device), tdepth.to(self.device))
                        .argmax(dim=1)
                        .item()
                    )
                    pre_rgb = rgb
                    pre_depth = depth

                obs, reward, done, info = self.env.step(action=action_sac)
                logger.info(
                    f"take action sac model {action_sac} ,reward {reward} , step {step} , "
                    f"done {done} , is collided {self.sim.previous_step_collided}"
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
    
    def rollout_transformer_v16(self, epoch, sac_model, logger):
        """
        v1_6: hybrid embedding (two-frame) + transformer policy over encoded sequence.
        """
        total_reward = 0.0
        total_spl = 0.0
        default_seq = int(getattr(self.config, "lstm_seq_len", 5))
        seq_len = int(getattr(self.config, "transformer_seq_len", default_seq))
        max_seq_len = int(getattr(self.config, "max_seq_len", seq_len))
        seq_len = max(1, min(seq_len, max_seq_len))
        max_steps = int(getattr(self.config, "MAX_STEPS", 200))

        for _ in tqdm(range(self.env._env.number_of_episodes), desc="onlinetest"):
            obs = self.env.reset()
            scene = self.env._env.current_episode.scene_id[-15:-4]
            episode_id = self.env._env.current_episode.episode_id
            logger.info(f"{epoch} , scene is {scene}  , episodeid is {episode_id}")

            done = False
            step = 0
            episode_reward = 0.0
            info = {"spl": 0.0, "distance_to_goal": -1.0}
            pre_rgb = None
            pre_depth = None
            state_queue = deque(maxlen=seq_len)

            while (not done) and (step < max_steps):
                with torch.no_grad():
                    rgb = torch.from_numpy(obs["rgb"]).float() / 255.0
                    depth = torch.from_numpy(obs["depth"]).float()
                    audio = self._get_audio_tensor(obs)
                    if pre_rgb is None:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    state = (
                        self.hybirdmodel.embedding_forward(
                            audio.to(self.device),
                            trgb.to(self.device),
                            tdepth.to(self.device),
                        )
                        .detach()
                        .cpu()
                    )
                    state_queue.append(state)

                    while len(state_queue) < seq_len:
                        state_queue.appendleft(torch.zeros_like(state))
                    seq_state = torch.stack(list(state_queue), dim=0).unsqueeze(0)
                    action_sac = self._to_env_action(
                        sac_model.get_action(seq_state.to(self.device), eval=True)
                    )
                    action_hybird = int(
                        self.hybirdmodel(
                            audio.to(self.device), trgb.to(self.device), tdepth.to(self.device)
                        )
                        .argmax(dim=1)
                        .item()
                    )
                    pre_rgb = rgb
                    pre_depth = depth

                obs, reward, done, info = self.env.step(action=action_sac)
                logger.info(
                    f"take action sac model {action_sac} ,reward {reward} , step {step} , "
                    f"done {done} , is collided {self.sim.previous_step_collided}"
                )
                episode_reward += float(reward)
                step += 1

            logger.info(
                f"episode is done , distance_to_goal is {info['distance_to_goal']}, "
                f"spl is {info['spl']} \nsumreward is {episode_reward}"
            )
            total_reward += episode_reward
            total_spl += float(info.get("spl", 0.0))

        eps = float(self.env._env.number_of_episodes)
        return total_reward / eps, total_spl / eps


    def rollout(self , epoch , sac_model , logger):
        if self.config.model == "v1":
            return self.rollout_two_frame(epoch , sac_model , logger)
        elif self.config.model in ("v1_3", "v1_4"):
            return self.rollout_hybrid_offline(epoch , sac_model , logger)
        elif self.config.model == 'v2':
            return self.rollout_two_frame(epoch , sac_model , logger)
        elif self.config.model == 'v4':
            return self.rollout_lstm(epoch , sac_model , logger)
        elif self.config.model == 'v5':
            return self.rollout_lstm_attention(epoch , sac_model , logger)
        elif self.config.model == 'v1_5':
            return self.rollout_lstm_v15(epoch, sac_model, logger)
        elif self.config.model == 'v1_6':
            return self.rollout_transformer_v16(epoch, sac_model, logger)
        raise ValueError(f"Unsupported online test model: {self.config.model}")
    
