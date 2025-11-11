# test the sim navmash point
from env import Env
from configs.default import get_config
import os
from habitat.utils.visualizations import maps
from PIL import Image
from soundspaces.utils import load_metadata
from utils.visualizations import get_topdown_map , draw_sound

config = get_config()
env = Env(config=config)
for _ in range(env._env.number_of_episodes):
    env.reset()
    sim = env._env.sim
    top_down_map = get_topdown_map(sim)
    graphs = sim.graph.nodes()
    points = []
    for node_point in graphs:
        point = graphs[node_point]['point']
        if sim.pathfinder.is_navigable(point):
            points.append(point)

    draw_sound(sim , points , top_down_map , maps.MAP_VIEW_POINT_INDICATOR)
    Image.fromarray(top_down_map).save(f'./scene_nodes/{env._env.current_episode.scene_id[-15:-4]}.png')

