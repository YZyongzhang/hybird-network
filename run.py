from env import Env
from configs.default import get_config
from utils.log import logger
config = get_config()

if __name__ == "__main__":
    task_config = config.TASK_CONFIG
    
    if task_config.COLLECT.OPEN :
        from col import COLLECTER
        from run import Collect
        import torch
        from network import HybirdNetwork
        env = Env(config=config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybirdNetwork().to(device)
        model.load_state_dict(torch.load(task_config.COLLECT.COLLECT_CKPT))
        model.eval()
        collecter = COLLECTER(config , env , model = model)
        logger.info(f"collect {task_config.COLLECT.TYPE} beggining")
        Collect(collecter=collecter)
    
    elif task_config.LMDB.OPEN :
        loadlmdb_config = task_config.LMDB
        from train import LoadLmdb
        if loadlmdb_config.TYPE == "HybirdNetwork":
            LoadLmdb.load_pt(task_config.LMDB.RAW_DATA_PATH , task_config)
        elif loadlmdb_config.TYPE == "offline":
            from network import HybirdNetwork
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = HybirdNetwork().to(device)
            model.load_state_dict(torch.load(loadlmdb_config.CKPT))
            model.eval()
            LoadLmdb.load_offline(loadlmdb_config.RAW_DATA_PATH , model= model, config=task_config)
    
    elif task_config.TRAIN.OPEN:
        train_config = task_config.TRAIN
        offline_config  = train_config.OFFLINE
        if train_config.TYPE == "HybirdNetworkAudio":
            from network import AudioCRNN
            from run import Train
            from train import HybirdNetworkAudioTrain
            model = AudioCRNN()
            Train(model=model , trainer=HybirdNetworkAudioTrain , config=train_config)
        elif train_config.TYPE == "HybirdNetwork":
            from network import HybirdNetwork
            from run import Train
            from train import HybirdNetworkTrain
            model = HybirdNetwork()
            Train(model=model , trainer=HybirdNetworkTrain , config=train_config)
        elif train_config.TYPE == "OfflineRL":
            from network import SAC_model
            from run import Train
            from train import OfflineTrain
            from train import OnlineTest
            from network import HybirdNetwork
            import torch
            env = Env(config)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = HybirdNetwork().to(device)
            model.load_state_dict(torch.load(offline_config.ONLINE_CKPT))
            model.eval()
            
            online_test = OnlineTest(env=env , hybirdmodel=model)
            
            state_dim = offline_config.state_dim
            action_dim = offline_config.action_dim
            hidden_dim = offline_config.hidden_dim
            lr = offline_config.lr
            target_entropy = offline_config.target_entropy  
            tau = offline_config.tau
            gamma = offline_config.gamma

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
                beta=1,
                device=device
            )
            if offline_config.LOAD_PATH:
                sac_model.load_state_dict(torch.load(offline_config.MODEL_PATH))
                sac_model.train()
            Train(model=sac_model , trainer=OfflineTrain  , config= offline_config , online_test = online_test )
            
            
        
        
