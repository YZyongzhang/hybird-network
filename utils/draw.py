from matplotlib import pyplot as plt
import math
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
import magnum as mn
import numpy as np
import quaternion
from PIL import Image
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
class Draw:
    def __init__(self):
        pass

    def convert_points_to_topdown(self , pathfinder, points, meters_per_pixel=0.01):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # print(point)
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown

    def display_path_map(self, topdown_map, key_points=None, path=None, image_filename=None):
        output_dir = './output_images'
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        ax.axis("off")
        ax.imshow(topdown_map)

        if path is not None:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            print(path_x)
            ax.plot(path_x, path_y, marker="o", markersize=5, color="darkblue", alpha=0.7, label="Path", linewidth=5)
            ax.plot(path_x[-1], path_y[-1], marker='o', markersize=5, color='blue', alpha=0.7, label="final_point")

        if key_points is not None:
            for point in key_points:
                ax.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color="red", label="Key Point")

        # 保存图片（可选）
        if image_filename:
            fig.savefig(image_filename, bbox_inches='tight', pad_inches=0)

        # 将绘制的图像保存成 numpy 数组
        canvas.draw()
        img_np = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # (H, W, 3)

        plt.close(fig)  # 关闭图像释放内存

        return img_np
    # display a topdown map with matplotlib
    def display_map(self , topdown_map, key_points=None):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(topdown_map)
        # plot points on map
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        plt.show(block=False)


    def get_td_map(self , pathfinder, meters_per_pixel=0.01, vis_points=None):

        height = pathfinder.get_bounds()[0][1]
        xy_vis_points = None

        if vis_points is not None:
            xy_vis_points = self.convert_points_to_topdown(
                pathfinder, vis_points, meters_per_pixel
            )

        hablab_topdown_map = maps.get_topdown_map(
            pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]

        return hablab_topdown_map, xy_vis_points

    def add_agent_pos_angle(self , pathfinder, top_down_graph, agent_pos, agent_q ,sim):
        grid_dimensions = (top_down_graph.shape[0], top_down_graph.shape[1])
        agent_grid_pos = maps.to_grid(
            agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=pathfinder
        )
        agent_forward = utils.quat_to_magnum(
            sim.agents[0].get_state().rotation
        ).transform_vector(mn.Vector3(0, 0, -1.0))
        agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        # draw the agent and trajectory on the map
        maps.draw_agent(
            top_down_graph, agent_grid_pos, agent_orientation, agent_radius_px=16
        )
        # maps.draw_agent(
        #     top_down_graph, grid_pos, show_angle, agent_radius_px=16
        # )

        return top_down_graph
    def show_graph(self , env):
        sim = env._sim
        source_pos = env.get_source_pos()
        agent_pos = env.get_agent_pos()
        vis_points = source_pos
        x, y = self.get_td_map(env._sim.pathfinder, vis_points=vis_points)
        agent_r = env.get_agent_rotation()

        for agent_id in range(env._num_agents):
            x = self.add_agent_pos_angle(env._sim.pathfinder, x, agent_pos[agent_id], agent_r[agent_id] ,sim)
        
        self.display_map(x,y)
    def show_path_graph(self , env , sound , agent ,image_filename):
        source_pos = sound
        agent_point = agent
        x, s = self.get_td_map(env._sim.pathfinder, vis_points=source_pos)
        x, a = self.get_td_map(env._sim.pathfinder, vis_points=agent_point)
        
        grid_dimensions = (x.shape[0], x.shape[1])
        agent_grid_pos = maps.to_grid(
            agent_point[-1][2], agent_point[-1][0], grid_dimensions, pathfinder=env._sim.pathfinder
        )
        agent_forward = utils.quat_to_magnum(
            np.quaternion(1, 0, 0, 0)
        ).transform_vector(mn.Vector3(0, 0, -1.0))
        agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        maps.draw_agent(
            x, agent_grid_pos, agent_orientation, agent_radius_px=16
        )
        
        self.display_path_map(x,s,a,image_filename)
    def draw(self, sim ,path):
        meters_per_pixel = 0.025
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        height = scene_bb.y().min
        top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
        recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        trajectory = [
                    maps.to_grid(
                        path_point_[2],
                        path_point_[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
                    for path_point_ in path
                ]
        grid_tangent = mn.Vector2(
                    trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                )
        path_initial_tangent = grid_tangent / grid_tangent.length()
        initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
        maps.draw_path(top_down_map, trajectory)
        maps.draw_agent(
        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
        )
        return top_down_map
    # def optim_point(self,points):
    #     # point is agent path point list(np.array()) - > list(np.array())
    #     # get the frist point to compute angle 
    #     # result is list(np.array())
    #     result_point = list()
    #     begin_point = points[0]
    #     for p in points:
    #         angle = self.compute_angle(begin_point , p)
    #         if angle
def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None
                
class muti_env_draw:
    def __init__(self):
        self.meters_per_pixel = 0.1
    def show_graph(self,env):
        sim = env._get_sim()
        height = sim.pathfinder.get_bounds()[0][1]
        # height = 1
        sim_topdown_map = sim.pathfinder.get_topdown_view(self.meters_per_pixel, height)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=self.meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        print("Displaying the raw map from get_topdown_view:")
        self.display_map(sim_topdown_map)
        print("Displaying the map from the Habitat-Lab maps module:")
        self.display_map(hablab_topdown_map)

        # easily save a map to file:
        # map_filename = os.path.join(output_path, "top_down_map.png")
        # imageio.imsave(map_filename, hablab_topdown_map)
    def convert_points_to_topdown(self, pathfinder, points, meters_per_pixel):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown
    def display_map(self, topdown_map, key_points=None):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        topdown_map = np.rot90(topdown_map, k=1) 
        plt.imshow(topdown_map)
        # plot points on map
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        plt.show(block=False)