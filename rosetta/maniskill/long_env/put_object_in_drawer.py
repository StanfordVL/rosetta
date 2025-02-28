from typing import Any, Dict, Union, List
import numpy as np
import torch
import sapien.core as sapien

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.geometry.geometry import transform_points
from collections import defaultdict

@register_env("PutObjectInDrawer", max_episode_steps=3000)
class PutObjectInDrawer(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # Asset configuration and constants
    DRAWER_ASSET_ID = "partnet_mobility_cabinet"
    handle_types = ["prismatic"]  # We are interested in prismatic joints (drawers)
    min_open_frac = 0.6  # Fraction of the drawer's range to be open
    skill_config={
        "home_pos": (0.0,0.0,0.5),
    }
    apple_model_id = "ycb:013_apple"
    soup_model_id = "ycb:005_tomato_soup_can"
    def __init__(self, stage=0, *args, robot_uids="panda", robot_init_qpos_noise=0.02, drawer_id=1016, **kwargs):
        self.stage = stage
        self.cur_stage = 0
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.workspace_x = [-0.7, -0.1]  # Adjusted workspace X range
        self.workspace_y = [-0.57, 0.2]
        self.workspace_z = [0.01, 0.47]
        self.drawer_id = drawer_id
        self.reward_components = ["success", "afford"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
            0: "pick",
            1: "place",
            2: "push",
        }

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.5, 0, 0.6], target=[-0.5, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.3, 1.0], [-0.5, -0.2, 0.35])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        # Build ground plane
        self.ground = build_ground(self.scene)

        # Load the drawer
        self._load_drawer()

        # Load YCB objects
        builder_apple = actors.get_actor_builder(self.scene, id=self.apple_model_id)
        self.apple = builder_apple.build_dynamic(name="apple")

        builder_soup = actors.get_actor_builder(self.scene, id=self.soup_model_id)
        self.soup_can = builder_soup.build_dynamic(name="soup_can")

    def _load_drawer(self):
        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self.drawer_id}"
        )
        cos_theta_over_2 = np.cos(-np.pi / 4)
        sin_theta_over_2 = np.sin(-np.pi / 4)

        # Adjusted drawer position and orientation
        cabinet_builder.initial_pose = sapien.Pose(
            p=[-0.5, -1.0, 0.5],  
            q=[cos_theta_over_2, 0, 0, sin_theta_over_2]  
        )
        cabinet = cabinet_builder.build(name=f"{self.drawer_id}-drawer")
        for joint in cabinet.get_joints():
            if joint.type[0] in self.handle_types and joint.active_index is not None:
                self.drawer = cabinet
                self.drawer_joint = joint
                print(f"Loaded drawer model_id: {self.drawer_id}")
                print(f"Found drawer joint: {joint.get_name()}")
                break

    def _after_reconfigure(self, options):
        qlimits = self.drawer.get_qlimits()  
        num_envs = qlimits.shape[0]
        env_idx = torch.arange(num_envs, device=self.device)
        qmin = qlimits[env_idx, self.drawer_joint.active_index, 0]
        qmax = qlimits[env_idx, self.drawer_joint.active_index, 1]
        self.drawer_open_positions = qmin + (qmax - qmin) * self.min_open_frac

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Get apple's bounding box bounds correctly
            apple_bounds = self.apple.get_first_collision_mesh().bounding_box.bounds
            apple_height = (apple_bounds[1][2] - apple_bounds[0][2]) / 2  # Calculate height from bounds

            soup_can_bounds = self.soup_can.get_first_collision_mesh().bounding_box.bounds
            soup_can_height = (soup_can_bounds[1][2] - soup_can_bounds[0][2]) / 2  # Calculate height from bounds

            # Sample apple position
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = apple_height  # Use apple's actual height
            sampler = randomization.UniformPlacementSampler(bounds=[
                [self.workspace_x[0]+0.55, self.workspace_y[0]+0.5],
                [self.workspace_x[1], self.workspace_y[1]]
            ], batch_size=b)
            apple_xy = sampler.sample(0.04, 100)  # Adjust radius based on apple size if needed
            xyz[:, :2] = apple_xy
            self.apple.set_pose(Pose.create_from_pq(p=xyz.clone()))

            # Sample soup can position
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = soup_can_height  # Use soup can's actual height
            sampler = randomization.UniformPlacementSampler(bounds=[
                [self.workspace_x[0]+0.55, self.workspace_y[0]+0.3],
                [self.workspace_x[1], self.workspace_y[0]+0.4]
            ], batch_size=b)
            soup_can_xy = sampler.sample(0.04, 100)  # Adjust radius based on soup can size if needed
            xyz[:, :2] = soup_can_xy
            self.soup_can.set_pose(Pose.create_from_pq(p=xyz.clone()))

            # Reset drawer position
            qpos = self.drawer.get_qpos()
            drawer_active_index=self.drawer_joint.active_index[env_idx]
            qpos[env_idx, drawer_active_index] = self.drawer_open_positions[env_idx]
            self.drawer.set_qpos(qpos[env_idx].clone())

            # Initialize robot's joint positions
            qpos = np.array([0.0, 0, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])
            self.agent.robot.set_pose(sapien.Pose([-0.5, 0, 0]))
            self.agent.robot.set_qpos(qpos)
            self.object_list = {"apple": self.apple, "soup_can": self.soup_can}
        
    def compute_normalized_dense_reward(self, obs, action, info):
        # Normalize dense reward
        max_possible_reward = 30.0
        return self.compute_dense_reward(obs, action, info) / max_possible_reward
    
    def _get_obs_info(self):
        info= {}
        for name in self.object_list:
            info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
            info[f"{name}_pos"] = self.object_list[name].pose.p[0]
        
        info["stage"] = self.cur_stage
        info["gripper_pos"] = self.agent.tcp.pose.p[0]

        drawer_link = self.drawer_joint.get_child_link()
        info["drawer_handle_pos"] = drawer_link.pose.p.to(self.device)[0] + np.array([0.01, 0.37, -0.3])
        info["drawer_pos"] = drawer_link.pose.p.to(self.device)[0] + np.array([0.01, 0.2, -0.3])

        info["drawer_open_offset"] = (info["drawer_handle_pos"][1] - np.array([0.5, -0.63,  0.2]))[1]
        return info

    def evaluate(self):
        info = self._get_obs_info()

        def stage0_success(info):
            return info[f"is_apple_grasped"]
        
        def stage1_success(info):
            # Check if apple is in drawer
            abs_diff_xy = torch.abs(info["apple_pos"][:2] - info["drawer_pos"][:2])

            # Check if x and y differences are within tolerance
            within_xy = (abs_diff_xy <= 0.1).all(dim=-1)

            # Check if z position is within valid range
            within_z = (info["apple_pos"][2] >= 0.1) & (info["apple_pos"][2] <= 0.4)

            # Combine conditions
            is_apple_in_drawer = within_xy & within_z
            return is_apple_in_drawer & (~info["is_apple_grasped"])

        info["stage0_success"] = stage0_success(info)
        info["stage1_success"] = stage1_success(info)
        info["success"] = torch.tensor(False)
        if self.cur_stage==1:
            info["success"] = info["stage1_success"]

        return info

    def get_obs(self, info: Dict = None):
        if info is None:
            info = self.get_info()
        obs = []
        for name in self.object_list:
            obs += info[f"{name}_pos"].flatten().tolist()

        for name in self.object_list:
            obs += info[f"is_{name}_grasped"].flatten().tolist()
        
        obs += [self.cur_stage]
        return torch.tensor([obs], device = self.device, dtype = torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        reward_components = dict((k, 0.0) for k in self.reward_components)
        current_selected_action = np.argmax(action[:len(self.task_skill_indices.keys())])

        if current_selected_action in [0, 1]:
            current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
            current_selected_pos_2 = None
        else:
            current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
            current_selected_pos_2 = action[len(self.task_skill_indices.keys())+3:len(self.task_skill_indices.keys())+6]

        def stage_0_reward():
            target_action = 0
            target_pos_1 = prev_info["apple_pos"].copy()
            target_pos_2 = None

            if current_selected_action == target_action:
                reward = -np.tanh(np.linalg.norm(current_selected_pos_1 - target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage0_success"]:
                reward_components["success"] = 10
            return reward_components

        def stage_1_reward():
            target_action = 1  # place
            target_pos_1 =  prev_info["drawer_pos"] + np.array([0,0.05,0.25])
            target_pos_2 = None

            if current_selected_action == target_action:
                reward = -np.tanh(np.linalg.norm(current_selected_pos_1 - target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage1_success"]:
                reward_components["success"] = 10
        
            return reward_components

        if self.cur_stage==0:
            reward = stage_0_reward()
        elif self.cur_stage==1:
            reward = stage_1_reward()
        
        
        cur_before = self.cur_stage
        # move to next stage if success
        if (self.cur_stage == 0) and cur_info["stage0_success"]:
            self.cur_stage = 1
        elif (self.cur_stage == 1) and cur_info["stage1_success"]:
            self.cur_stage = 2

        print("stage:",cur_before, "->", self.cur_stage)
        return reward

    def reset(self, **kwargs):
        self.cur_stage = 0
        return super().reset(**kwargs)