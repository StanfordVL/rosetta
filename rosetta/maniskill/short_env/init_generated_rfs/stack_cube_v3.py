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

@register_env("StackCube-v3", max_episode_steps=100)
class StackCubeEnv_v3(BaseEnv):

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
        self.cubeD = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeD"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1

            region1 = [[-0.1, -0.2], [0.1, 0.0]]
            region2 = [[-0.1, 0.0], [0.1, 0.2]]
            
            sampler1 = randomization.UniformPlacementSampler(bounds=region1, batch_size=b)
            sampler2 = randomization.UniformPlacementSampler(bounds=region2, batch_size=b)
            
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            
            cubeA_xy = xy + sampler1.sample(radius, 100)
            cubeB_xy = xy + sampler1.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler2.sample(radius, 100)
            cubeD_xy = xy + sampler2.sample(radius, 100, verbose=False)

            # Initialize positions and orientations for cubes A and B
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

            # Initialize positions and orientations for cubes C and D
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeD_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeD.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

     def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                cubeD_pose=self.cubeD.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeD_pos=self.cubeD.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeC_to_cubeD_pos=self.cubeD.pose.p - self.cubeC.pose.p,
            )
        return obs

    def evaluate(self: StackCubeEnv_v3) -> Dict[str, bool]:
        """
        Evaluate the current state of the environment to determine if the task is successful.
        """
        success = (
            self.cubeA.pose.p[2] > self.cubeB.pose.p[2] and 
            torch.allclose(self.cubeA.pose.p[:2], self.cubeB.pose.p[:2], atol=0.02) and
            self.cubeC.pose.p[2] > self.cubeD.pose.p[2] and
            torch.allclose(self.cubeC.pose.p[:2], self.cubeD.pose.p[:2], atol=0.02)
        )
        return {
            "success": success,
            "cubeA_on_cubeB": self.cubeA.pose.p[2] > self.cubeB.pose.p[2] and torch.allclose(self.cubeA.pose.p[:2], self.cubeB.pose.p[:2], atol=0.02),
            "cubeC_on_cubeD": self.cubeC.pose.p[2] > self.cubeD.pose.p[2] and torch.allclose(self.cubeC.pose.p[:2], self.cubeD.pose.p[:2], atol=0.02),
        }

    def compute_dense_reward(self: StackCubeEnv_v3, obs: Any, action: torch.Tensor, info: Dict[str, bool]) -> float:
        """
        Compute the dense reward based on the current observation, action, and evaluation info.
        """
        reward = 0.0

        if action[0] > action[1]:
            # Current action is "grasp"
            # Reward for being close to any cube
            tcp_pos = obs['tcp_pose'][:3]
            cube_positions = [obs['cubeA_pose'][:3], obs['cubeB_pose'][:3], obs['cubeC_pose'][:3], obs['cubeD_pose'][:3]]
            distances = [torch.norm(tcp_pos - cube_pos) for cube_pos in cube_positions]
            reward += max(1.0 - min(distances), 0.0)  # Reward for being close to a cube

            # Additional reward for grasping a cube
            if self.agent.is_grasping():
                reward += 1.0

        if action[0] < action[1]:
            # Current action is "place"
            # Reward for being above any other cube
            tcp_pos = obs['tcp_pose'][:3]
            cube_positions = [obs['cubeB_pose'][:3], obs['cubeD_pose'][:3]]
            distances = [torch.norm(tcp_pos[:2] - cube_pos[:2]) for cube_pos in cube_positions]
            reward += max(1.0 - min(distances), 0.0)  # Reward for being above a cube

            # Additional reward for placing a cube on another cube
            if info['cubeA_on_cubeB']:
                reward += 2.0
            if info['cubeC_on_cubeD']:
                reward += 2.0

        # Final success reward
        if info['success']:
            reward = 8.75

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8.75
