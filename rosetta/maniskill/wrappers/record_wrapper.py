import gymnasium as gym
import torch
import sapien.physx as physx
#from feedback_to_reward.maniskill.utils.primitive_skills_cpu import PrimitiveSkillDelta
from feedback_to_reward.maniskill.primitive_skills.primitive_skills_cpu import PrimitiveSkillDelta
import numpy as np
from mani_skill.utils import common
import sys
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils import common, gym_utils
from copy import deepcopy
import os
import json
from PIL import Image


class RecordWrapper(gym.Wrapper):
    """
    This wrapper wraps any maniskill CPUEnv and records the trajectory
    """
    def __init__(self, env: gym.Env,
                 record_dir=None,
                 sample_freq = 1,
                 video_fps: int = 30,
                 **kwargs):
        super().__init__(env)
        self.record_dir = record_dir

        self.elapsed_step = 0
        self.video_fps=video_fps
        self.sample_freq = sample_freq
        self.render_images=[]
        self._video_steps=0
        self.episode_id=0
        self.is_params_scaled = True
        self.traj = []

        if self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True)
            os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}"), exist_ok=True)
            os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}", "frames"), exist_ok=True)
    
    def step(self, action,**kwargs):
        self.elapsed_step+=1
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        if self.record_dir is not None:

            
            self.record_image()
            if (self.elapsed_step % self.sample_freq) == 0:
                state_info = {}

                for name in self.object_list:
                    if "goal" in name:
                        state_info[f"{name}_pos"] = self.object_list[name].pose.p[0]
                    else:
                        state_info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
                        state_info[f"{name}_pos"] = self.object_list[name].pose.p[0]
                
                state_info["gripper_pos"] = self.agent.tcp.pose.p[0]
            
                self.traj.append(state_info)
                img = self.render_images[-1]
                
                img = (img).astype(np.uint8)
                img_path = os.path.join(self.record_dir, f"episode_{self.episode_id}", "frames", f"{len(self.render_images)}.png")
                Image.fromarray(img).save(img_path)


 
        return obs, reward, terminated, truncated, info

    def reset(self,seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # TODO: Make sure this is not a bug
        # obs, reward, terminated, truncated, info = super().step(action_ll)
        self.elapsed_step=0

        state_info = {}

        for name in self.object_list:
            if "goal" in name:
                state_info[f"{name}_pos"] = self.object_list[name].pose.p[0]
            else:
                state_info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
                state_info[f"{name}_pos"] = self.object_list[name].pose.p[0]
        
        state_info["gripper_pos"] = self.agent.tcp.pose.p[0]

        # record the trajectory
        if self.record_dir is not None:
            self.flush_video()
            if len(self.traj) > 0:
                self.traj = convert_arrays_to_lists(self.traj)
                try:
                    with open(os.path.join(self.record_dir, f"episode_{self.episode_id}", "trajectory.json"), "w") as f:
                        json.dump(self.traj, f, indent=4) # HACK to handle np.float32 objects
                except Exception as e:
                    print(f"Failed to save trajectory: {e}")
                    print(self.traj)
                self.episode_id += 1
                os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}"), exist_ok=True)
                os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}", "frames"), exist_ok=True)
            
            self.record_image()
            img = self.render_images[-1]
            img = (img).astype(np.uint8)
            img_path = os.path.join(self.record_dir, f"episode_{self.episode_id}", "frames", f"{self._video_steps}.png")
            Image.fromarray(img).save(img_path)
            self.traj = [state_info]
            
        return obs, info
    
    def record_image(self):
        img=super().render().cpu().numpy()[0]
        self.render_images.append(img)
            
    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            if name is None:
                video_name = "{}".format(self.episode_id)
                if suffix:
                    video_name += "_" + suffix
            else:
                video_name = name
            images_to_video(
                self.render_images,
                str(os.path.join(self.record_dir, f"episode_{self.episode_id}")),
                video_name=video_name,
                fps=self.video_fps,
                verbose=verbose,
            )
        self._video_steps = 0
        self.render_images = []


def convert_arrays_to_lists(data):
    """
    Recursively traverse a nested dictionary (or list) and convert numpy arrays to lists.
    
    Args:
        data (dict, list, any): The input nested dictionary, list, or value.
        
    Returns:
        The transformed structure with numpy arrays converted to lists.
    """
    if isinstance(data, dict):
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_arrays_to_lists(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data