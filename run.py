from env import Env
from configs.default import get_config
from utils.log import logger
config = get_config()

if __name__ == "__main__":
    task_config = config.TASK_CONFIG
    
    if task_config.COLLECT.OPEN :
        from col import COLLECTER
        from run import Collect
        env = Env(config=config)
        collecter = COLLECTER(config , env)
        logger.info(f"collect {task_config.COLLECT.TYPE} beggining")
        Collect(collecter=collecter)
    
    elif task_config.LMDB.OPEN :
        from train import LoadLmdb
        LoadLmdb.load_pt(task_config.LMDB.RAW_DATA_PATH , task_config)
    
    elif task_config.TRAIN.OPEN:
        train_config = task_config.TRAIN
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
            pass
    
        
        
