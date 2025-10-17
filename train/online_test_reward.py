import sys
 
from module import *
from network.foundation_model import Network

random.seed(int(time.time()))


class Actor:
    def __init__(self , model):
        self.step = 0 
        foundation_model_path = '/home/getuanhui/project/finnal_exp/experiment/train/ckpt/model_epoch_1000.pth'
        self.foundation_model = Network()
        self.foundation_model.load_state_dict(torch.load(foundation_model_path))
        self.foundation_model.eval()
        self.every_step_point = list()

        self.model = model
    def reset(self):
        self._idx = 0

    def act(self, env , obs):
        ret = list()

        
        num_agents = env.get_num_agents()
        for agent_id in range(num_agents):
            import pdb;pdb.set_trace()
            
            audio = obs[agent_id]['audio']
            visual = obs[agent_id]['rgb']
            encoder = self.foundation_model(audio , visual)
            action = nn.argmax(self.mode(encoder))
            print("agent", agent_id, action)

            logging.info(f"excute action {action}")
            ret.append(
                {
                    "rl_pred": act_id,
                    "lstm_h": None,
                    "lstm_c": None,
                }
            )
        self._idx += 1

        return ret
    
    def agent_step(self,env , obs):
        self.reset()
        self.step += 1

        all_done_list = list()
        all_r_list = list()
        
        while True:
            self.every_step_point.append(env.get_agent_pos()[0])

            rl_output_list = self.act(env , obs)
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

            
        result = [seq_list , all_done_list , self.every_step_point]
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
    if env._if_find_path_isNone(agent_id=0 , goal_pos=[np.array(audio_pos)] , agent_pos=[np.array(agent_pos)]): # 单个agent ，后续多agent可以使用for i in agentnum
        
        logging.error(f"env error ! raise habitat_sim.errors.GreedyFollowerError this episode {episode['episode_id']} jump")
        env.__clear__()
        return None
    
    obs = env.reset(audio_pos = [np.array(audio_pos)], agent_pos = [np.array(agent_pos)])

    print(f"reset_env success . init scene {episode['scene_id']} and sound {info['sound']}")
    logging.info(f"reset_env success . init scene {episode['scene_id']} and sound {info['sound']}")

    return env , obs

def collect(episodes_dict , collect_name , model):
    for step_id , episode_dict  in enumerate(episodes_dict):
        # create env config about scene and sound pos

        # check episode if exit in collect dir
        episode_id = episode_dict['episode_id']

        print(f"this begin collect the episode id {episode_id}")
        logging.info(f"this begin collect the episode id {episode_id}")


        env ,obs = reset_env(episode_dict)

        if not env:
            continue

        # init actor
        actor = Actor(model)

        result_lists , sum_reward = actor.agent_step(env , obs)

        os.makedirs(f"/home/getuanhui/project/finnal_exp/experiment/data/{collect_name}" , exist_ok= True)
        path = os.path.join(f"/home/getuanhui/project/finnal_exp/experiment/data/{collect_name}",  f"rl_step_{step_id}_episode_{episode_id}.pkl")

        with open(path, "wb") as f:
            pickle.dump(result_lists, f)

        env.__clear__()
        actor.every_step_point.clear()


   
if  __name__== "__main__":



    print("collect data begin , please input this collect name or usage .....")
    collect_name = input("input here:")
    logging.basicConfig(
        filename=f"/home/getuanhui/project/finnal_exp/collect/log/{collect_name}.log",
        level=logging.INFO,
        filemode='a', 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    collect(train_config['episodes'], collect_name)
