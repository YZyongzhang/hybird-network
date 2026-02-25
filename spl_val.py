import torch
from run import Train
from train import OfflineTrainBuffer , OfflineTrain
from train import OnlineTest
from network import HybirdNetwork
from env import Env
from network import SAC_model

from configs.default import get_config
from utils.log import logger
config = get_config()
task_config = config.TASK_CONFIG
train_config = task_config.TRAIN
offline_config  = train_config.OFFLINE

# HybirdNetwork_path = "compare/ckpt/hybirdnetwork_RGBD_two_frame/model_epoch_100.pth"
HybirdNetwork_path = 'hu/heard/hybird_ckpt/model_epoch_100.pth'
sac_model_path = './offlineRL/ckpt/finnal_weak/26/sac_2level_model_1000.pth'


env = Env(config=config)
hybird = HybirdNetwork().to('cuda')
hybird.load_state_dict(torch.load(HybirdNetwork_path))
hybird.eval()
state_dim = offline_config.state_dim
action_dim = offline_config.action_dim
hidden_dim = offline_config.hidden_dim
lr = offline_config.lr
target_entropy = offline_config.target_entropy  
tau = offline_config.tau
gamma = offline_config.gamma
beta = offline_config.beta
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sac_model = SAC_model(
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim,
    actor_lr=lr,
    critic_lr=lr,
    alpha_lr=lr,
    target_entropy=target_entropy,
    tau=tau,
    gamma=gamma,
    beta=beta,
    device=device
)
# sac_model.load_state_dict(torch.load(sac_model_path))
sac_model.eval()
online = OnlineTest(env=env,hybirdmodel=hybird , config=offline_config)
reward , spl = online.rollout('sac',sac_model , logger)
logger.info(f"result is reward {reward} , spl is {spl}")