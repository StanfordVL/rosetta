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
@register_env("AlignCube-v1", max_episode_steps=150)
class AlignCubeEnv_v1(BaseEnv):

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

    def evaluate(self) -> Dict[str, bool]:
        """
        Evaluates the current state of the environment and returns a dictionary
        with boolean values indicating whether certain conditions are met.
        """
        info = {}
        # Check if each cube is in the correct position relative to each other
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        cubeC_pos = self.cubeC.pose.p

        info['cubeA_left_of_cubeB'] = cubeA_pos[0] < cubeB_pos[0]
        info['cubeB_left_of_cubeC'] = cubeB_pos[0] < cubeC_pos[0]

        # Check if the cubes are aligned in a straight line (within a tolerance)
        tolerance = 0.02
        info['cubes_aligned'] = (
            abs(cubeA_pos[1] - cubeB_pos[1]) < tolerance and
            abs(cubeB_pos[1] - cubeC_pos[1]) < tolerance
        )

        return info

    def compute_dense_reward(self: AlignCubeEnv_v1, obs: Any, action: torch.Tensor, info: Dict[str, bool]) -> float:
        """
        Compute the dense reward for the task of lining up blocks A, B, and C in order A-B-C left to right.
        """
        reward = 0.0

        # Extract positions of the cubes from the observation
        cubeA_pos = obs['cubeA_pose'][:3]
        cubeB_pos = obs['cubeB_pose'][:3]
        cubeC_pos = obs['cubeC_pose'][:3]

        # Define the desired positions for the cubes
        desired_cubeA_pos = torch.tensor([-0.05, 0.0, 0.02])
        desired_cubeB_pos = torch.tensor([0.0, 0.0, 0.02])
        desired_cubeC_pos = torch.tensor([0.05, 0.0, 0.02])

        # Calculate the distances to the desired positions
        distA = torch.norm(cubeA_pos - desired_cubeA_pos)
        distB = torch.norm(cubeB_pos - desired_cubeB_pos)
        distC = torch.norm(cubeC_pos - desired_cubeC_pos)

        # Reward for aligning cubes in the correct order
        if action[0] > action[1]:  # Grasp action
            # Encourage grasping the cubes close to their desired positions
            if self.agent.is_grasping(self.cubeA):
                reward += 1.0 / (1.0 + distA)
            elif self.agent.is_grasping(self.cubeB):
                reward += 1.0 / (1.0 + distB)
            elif self.agent.is_grasping(self.cubeC):
                reward += 1.0 / (1.0 + distC)

        elif action[0] < action[1]:  # Place action
            # Encourage placing the cubes in the correct order
            if self.agent.is_grasping(self.cubeA):
                reward += 1.0 / (1.0 + distA)
            elif self.agent.is_grasping(self.cubeB):
                reward += 1.0 / (1.0 + distB)
            elif self.agent.is_grasping(self.cubeC):
                reward += 1.0 / (1.0 + distC)

        # Additional reward for successfully aligning the cubes in the correct order
        if info["success"]:
            reward += 2.5 # Boost for completing the task

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 2.5
