from env import Env
from configs.default import get_config
from utils.log import logger
config = get_config()

if __name__ == "__main__":
    task_config = config.TASK_CONFIG
    
    if task_config.COLLECT :
        from col import COLLECTER
        from run import Collect
        env = Env(config=config)
        collecter = COLLECTER(config , env)
        logger.info(f"collect {task_config.COLLECT.TYPE} beggining")
        Collect(collecter=collecter)
    elif task_config.TRAIN.TYPE == "foundation":
        from network import Network
        from run import Train
        model = Network()
        Train(model=model , config=config)
        
        
