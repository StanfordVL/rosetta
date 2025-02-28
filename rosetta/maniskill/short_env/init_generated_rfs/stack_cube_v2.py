from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
@register_env("StackCube-v2", max_episode_steps=100000)
class StackCubeEnv_v2(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 1, 1], name="cubeC"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))


    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeC_to_cubeA_pos=self.cubeA.pose.p - self.cubeC.pose.p,
            )
        return obs

def evaluate(self: StackCubeEnv_v2) -> Dict[str, bool]:
    """
    Evaluate the current state of the environment and return a dictionary of reward-relevant questions.
    """
    info = {}
    
    # Check if cube A is on top of cube B
    cubeA_on_cubeB = (
        abs(self.cubeA.pose.p[0] - self.cubeB.pose.p[0]) < 0.02 and
        abs(self.cubeA.pose.p[1] - self.cubeB.pose.p[1]) < 0.02 and
        abs(self.cubeA.pose.p[2] - (self.cubeB.pose.p[2] + 0.04)) < 0.02
    )
    
    # Check if cube C is on top of cube A
    cubeC_on_cubeA = (
        abs(self.cubeC.pose.p[0] - self.cubeA.pose.p[0]) < 0.02 and
        abs(self.cubeC.pose.p[1] - self.cubeA.pose.p[1]) < 0.02 and
        abs(self.cubeC.pose.p[2] - (self.cubeA.pose.p[2] + 0.04)) < 0.02
    )
    
    # Task success is when both conditions are met
    info['cubeA_on_cubeB'] = cubeA_on_cubeB
    info['cubeC_on_cubeA'] = cubeC_on_cubeA
    info['success'] = cubeA_on_cubeB and cubeC_on_cubeA
    
    return info

def compute_dense_reward(self: StackCubeEnv_v2, obs: Any, action: torch.Tensor, info: Dict[str, bool]) -> float:
    """
    Compute the dense reward based on the current observation, action, and evaluation info.
    """
    reward = 0.0
    
    if action[0] > action[1]:
        # Current action is "grasp"
        if self.agent.is_grasping(self.cubeA) or self.agent.is_grasping(self.cubeB) or self.agent.is_grasping(self.cubeC):
            reward += 1.0  # Reward for successfully grasping a cube
    else:
        # Current action is "place"
        if self.agent.is_grasping(self.cubeA):
            # Reward for placing cube A on top of cube B
            if info['cubeA_on_cubeB']:
                reward += 10.0
            else:
                # Provide a small reward for moving cube A closer to cube B
                reward += max(0, 1.0 - torch.norm(self.cubeA.pose.p - self.cubeB.pose.p))
        
        if self.agent.is_grasping(self.cubeC):
            # Reward for placing cube C on top of cube A
            if info['cubeC_on_cubeA']:
                reward += 10.0
            else:
                # Provide a small reward for moving cube C closer to cube A
                reward += max(0, 1.0 - torch.norm(self.cubeC.pose.p - self.cubeA.pose.p))

    if info['success']:
        reward = 26.25  # Large reward for completing the task
        return reward
    
    return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 26.25