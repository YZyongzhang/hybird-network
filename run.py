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
    elif task_config.TRAIN.TYPE == "foundation":
        # from network import Network
        from network import AudioCRNN
        from run import Train
        model = AudioCRNN()
        Train(model=model , config=task_config.TRAIN)
    
        
        
