import sys
 
from module import *


random.seed(int(time.time()))


class Actor:
    def __init__(self):
        self.step = 0 
    
    def reset(self):
        self._idx = 0

    def act(self, env):
        ret = list()
        num_agents = env.get_num_agents()
        for agent_id in range(num_agents):
            if self._idx >= len(self.paths):
                action = "stop"
            else:
                action = self.paths[self._idx]
            print("agent", agent_id, action)

            logging.info(f"excute action {action}")

            act_id = env.action_str_2_id(action)

            ret.append(
                {
                    "rl_pred": act_id,
                    "lstm_h": None,
                    "lstm_c": None,
                }
            )
        self._idx += 1

        return ret
    
    def agent_step(self,env):
        self.reset()
        self.step += 1

        all_done_list = list()
        all_r_list = list()

        self.paths = env.get_shortest_action_list()[0]
        
        while True:
            rl_output_list = self.act(env)
            all_list = [env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            all_done_list.append(done_list)
            logging.info(f"reward is {r_list}")
            if all(done_list):
                for k, v in info_list[0].items():
                    logging.info(f"Env  {k}: {v}")
                break
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        logging.info(f"get reward {return_}")
        seq_list = list()
        seq_list += env.get()

            
        result = [seq_list , all_done_list]
        torch.cuda.empty_cache()
        return  result, return_


def reset_env(episode):

    # env data location 
    scene_path = '/home/getuanhui/project/sound-spaces/data/scene_datasets/mp3d/'
    audio_path = '/home/getuanhui/project/sound-spaces/data/sounds/1s_all_distractor/train'


    info = episode['info'] # to get the sound object
    scene = f"{scene_path}/{episode['scene_id']}" # get the env scene
    sound = f"{audio_path}/{info['sound']}.wav" # get the sound object

    agent_pos= episode['start_position'] # get the agent start_position
    audio_pos= episode['goals'][0]['position'] # get the sound position

    
    env_config['scene_dir'] = scene
    env_config['audio_dir'] = sound

    env = Env(env_config)


    # 防止出现环境中error的问题，出现问题就跳过
    if env._if_find_path_isNone(agent_id= 0 ):
        return None
    
    env.reset(audio_pos = [np.array(audio_pos)], agent_pos = [np.array(agent_pos)])

    return env

def collect(episodes_dict):
    episode_dict = episodes_dict[17]
    # create env config about scene and sound pos

    # check episode if exit in collect dir
    episode_id = episode_dict['episode_id']

    print(f"this begin collect the episode id {episode_id}")
    logging.info(f"this begin collect the episode id {episode_id}")


    env = reset_env(episode_dict)
    if env:
    # init actor
        actor = Actor()

        result_lists , sum_reward = actor.agent_step(env)

        env.__clear__()


   
if  __name__== "__main__":


    collect(train_config['episodes'])
