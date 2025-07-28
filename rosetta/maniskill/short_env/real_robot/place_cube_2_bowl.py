from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
import sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.utils import randomization

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

import matplotlib.pyplot as plt
import gymnasium as gym
@register_env("PlaceCube2Bowl", max_episode_steps=200)
class PlaceCube2BowlEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values - TODO Roger
    cube_half_size = 0.015
    model_id = "024_bowl"
    bowl1_scale = 1.0
    bowl2_scale = 1.0
    bowl1_floor_thickness = 0.01
    bowl2_floor_thickness = 0.01
    bowl1_base_half_height = 0.015
    bowl2_base_half_height = 0.015
    bowl1_half_thickness = 0.001
    bowl2_half_thickness = 0.001
        
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose_headon = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        pose_birdseye = sapien_utils.look_at([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        return [
            CameraConfig(
                "render_camera_headon", 
                pose=pose_headon, 
                width=512, 
                height=512, 
                fov=1, 
                near=0.01, 
                far=100
            )
        ]
        
    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([146, 228, 40, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.model_id}",
        )
        self.bowl1 = builder.build_dynamic(name=f"{self.model_id}_1")

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.model_id}",
        )
        self.bowl2 = builder.build_dynamic(name=f"{self.model_id}_2")
        #TODO scale the bowls

    def _after_reconfigure(self, options: dict):
        bowl1_collision_mesh = self.bowl1.get_first_collision_mesh()
        bowl2_collision_mesh = self.bowl2.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        bowl1_bounds = bowl1_collision_mesh.bounding_box.bounds
        bowl2_bounds = bowl2_collision_mesh.bounding_box.bounds
        
        self.bowl1_half_z = (bowl1_bounds[1][2] - bowl1_bounds[0][2]) / 2.
        self.bowl1_half_z = common.to_tensor(self.bowl1_half_z)
        self.bowl1_half_x = (bowl1_bounds[1][2] - bowl1_bounds[0][2]) * self.bowl1_scale / 2.
        self.bowl1_half_x = common.to_tensor(self.bowl1_half_x)
        self.bowl1_half_y = (bowl1_bounds[1][1] - bowl1_bounds[0][1]) * self.bowl1_scale / 2.
        self.bowl1_half_y = common.to_tensor(self.bowl1_half_y)

        self.bowl2_half_z = (bowl2_bounds[1][2] - bowl2_bounds[0][2]) / 2.
        self.bowl2_half_z = common.to_tensor(self.bowl2_half_z)
        self.bowl2_half_x = (bowl2_bounds[1][2] - bowl2_bounds[0][2]) * self.bowl2_scale / 2.
        self.bowl2_half_x = common.to_tensor(self.bowl2_half_x)
        self.bowl2_half_y = (bowl2_bounds[1][1] - bowl2_bounds[0][1]) * self.bowl2_scale / 2.
        self.bowl2_half_y = common.to_tensor(self.bowl2_half_y)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # init the cube in the first 1/4 zone along the x-axis (so that it doesn't collide the bowl)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0] # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0] # spanning all possible ys
            xyz[..., 2] = PlaceCube2BowlEnv.cube_half_size #self.cube_half_height # on the table
            q = [1, 0, 0, 0]
            cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(cube_pose)
            # self.cube.set_pose(Pose.create_from_pq(p=torch.zeros((b, 3)), q=[1, 0, 0, 0]))

            # TODO left off here - fix this for this setting
            pos1 = torch.zeros((b, 3))
            pos1[:, 0] = torch.rand((b, 1))[..., 0] * 0.05 + 0.05  # ensuring it's within [0.05, 0.1]
            # pos1[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1  # spanning all possible ys
            pos1[:, 1] = torch.rand((b, 1))[..., 0] * 0.05         # ensuring it's within [0, 0.05]
            pos1[:, 2] = PlaceCube2BowlEnv.bowl1_base_half_height  # on the table
            bowl_pose1 = Pose.create_from_pq(p=pos1, q=q)
            self.bowl1.set_pose(bowl_pose1)

            pos2 = torch.zeros((b, 3))
            pos2[:, 0] = pos1[:, 0] + 0.05   # slightly offset from bowl1 along the x-axis 
            pos2[:, 1] = pos1[:, 1] + 0.2       # fixed distance from bowl1 along the y-axis 
            pos2[:, 2] = PlaceCube2BowlEnv.bowl2_base_half_height  # on the table
            bowl_pose2 = Pose.create_from_pq(p=pos2, q=q)
            self.bowl2.set_pose(bowl_pose2)

    def evaluate(self):
        pos_cube = self.cube.pose.p
        pos_bowl = self.bowl1.pose.p
        offset = pos_cube - pos_bowl
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= 0.005
        )
        # Target position: center of the bowl, minus half z to get to the floor, then add on the bowl floor thickness for that offset, then add on the radius for the cube's offset
        z_flag = torch.abs(offset[..., 2] - self.bowl1_half_z + self.bowl1_floor_thickness + self.cube_half_size) <= 0.01
        is_cube_in_bowl = torch.logical_and(xy_flag, z_flag)
        is_cube_grasped = self.agent.is_grasping(self.cube)
        success = is_cube_in_bowl
        return {
            "is_cube_grasped": is_cube_grasped,
            "is_cube_in_bowl": is_cube_in_bowl,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            # is_grasped=info["is_cube_grasped"],
            is_grasped=self.agent.is_grasping(self.cube),
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bowl_pos=self.bowl1.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward_components = {}
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        cube_to_tcp_dist = torch.linalg.norm(tcp_pose - cube_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cube_to_tcp_dist))

        # grasp and place reward
        cube_pos = self.cube.pose.p
        bowl_pos = self.bowl1.pose.p
        bowl_inside_pos = self.bowl1.pose.p.clone()
        bowl_inside_pos[:, 2] = bowl_inside_pos[:, 2] - self.bowl1_half_z + PlaceCube2BowlEnv.bowl1_half_thickness + self.cube_half_size
        cube_to_bowl_inside_dist = torch.linalg.norm(bowl_inside_pos - cube_pos, axis=1)
        place_reward = (1 - torch.tanh(5.0 * cube_to_bowl_inside_dist))

        reward[info["is_cube_grasped"]] = (4 + place_reward)[info["is_cube_grasped"]]

        # # ungrasp and static reward
        # gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
        #     self.device
        # )
        # is_cube_grasped = info["is_cube_grasped"]
        # ungrasp_reward = (
        #     torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        # )
        # ungrasp_reward[~is_cube_grasped] = 16.0 # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        # v = torch.linalg.norm(self.cube.linear_velocity, axis=1)
        # av = torch.linalg.norm(self.cube.angular_velocity, axis=1)
        # static_reward = 1 - torch.tanh(v * 10 + av)
        # robot_static_reward = self.agent.is_static(0.2) # keep the robot static at the end state, since the sphere may spin when being placed on top

        # reward[info["is_cube_in_bowl"]] = (
        #     6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        # )[info["is_cube_in_bowl"]]
        
        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward