from env.RL_env import ENV as env
from env.RL_env import config
from col import COLLECTER
collecter = COLLECTER(config , env)
collecter.collect()