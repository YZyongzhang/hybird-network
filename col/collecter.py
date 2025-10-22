from habitat import Config
from ss_baselines.common.environments import AudioNavRLEnv
from col.register import CollectRegister
class COLLECTER:
    """
    collecter
    args:
        - greey collect
        - random collect
        - colliect collect
    """
    def __init__(self,config:Config,env:AudioNavRLEnv , **kwargs):
        self.env = env
        self.collect_config = config.TASK_CONFIG.COLLECT
        if self.collect_config.TYPE in ['greedy' , 'collided' , 'random' , 'angle' , 'offlineRL']:
            cls = CollectRegister.get(self.collect_config.TYPE)
            
            if self.collect_config.TYPE is 'offlineRL':
                self.collecter = cls(self.env , self.collect_config , model = kwargs['model'])
            else:
                self.collecter = cls(self.env , self.collect_config)
        else:
            raise ValueError(f"collect name error!")
    def collect(self):
        self.collecter.collect()