import gymnasium as gym
import inspect
from mani_skill.envs.sapien_env import BaseEnv
import difflib
import json

# add imports from task env to support reward manipulation
import torch
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

class RewardManipulator:
    def __init__(self,env_id=None,json_path=None,func_dict=None,env=None,sim_backend=None):
        self.new_method_dict=None
        if json_path is not None or func_dict is not None:
            self.new_method_dict=self.get_new_method_dict(json_path=json_path,new_method_dict=func_dict)
            if "Environment ID" in self.new_method_dict:
            # remove the environment id from the new method dict
                del self.new_method_dict["Environment ID"]
        if sim_backend is not None:
            self.sim_backend=sim_backend
        else:
            self.sim_backend="cpu"
        if env_id is not None:
            self.env_id=env_id
        else:
            self.env_id=self.extract_environment_id()
        
        self.env_cls=self.get_base_env_cls(env_id=self.env_id,env=env)
    
    def get_base_env_cls(self,env_id=None, env=None):
        assert env_id is not None or env is not None, "Either env_id or env must be provided."
        if env is not None:
            return self.get_base_env(env,BaseEnv).__class__
        else:
            env=gym.make(env_id,sim_backend=self.sim_backend,num_envs=1)
            env_cls=self.get_base_env(env,BaseEnv).__class__
            env.close()
            return env_cls

    def get_new_method_dict(self,json_path=None,new_method_dict=None):
        assert json_path is not None or new_method_dict is not None, "Either json_path or new_method_dict must be provided."
        if json_path is not None:
            with open(json_path, 'r') as file:
                new_method_dict=json.load(file)
        for k,v in new_method_dict.items():
            new_method_dict[k]=v.strip()
        return new_method_dict

    def extract_environment_id(self):
        return self.new_method_dict.get('Environment ID',None)

    def get_base_env(self,env, base_class):
        while not isinstance(env, base_class):
            if hasattr(env, 'env'):
                env = env.env
            else:
                source=f"Reached an object of type {type(env)} which does not have an 'env' attribute."
                return None
        return env

    def get_class_methods_source(self,cls):
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            try:
                source = inspect.getsource(method)
                methods[name] = source
            except OSError:
                print("Source code not available for method:", name) 
        return methods

    def get_diff_functions(self,new_function_dict, og_function_dict):
        def _get_diff(text1, text2):
            diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
            return '\n'.join(diff)

        common_functions = set(new_function_dict.keys()).intersection(set(og_function_dict.keys()))
        diff_functions = {}
        for function_name in common_functions:
            diff = _get_diff(og_function_dict[function_name],new_function_dict[function_name])
            diff_functions[function_name] = diff
        return diff_functions
    
    def change_function(self,functions_to_overwrite):
        assert self.new_method_dict is not None, "New method must not be none"
        exec_dict = {}
        for name,code_string in self.new_method_dict.items():
            if name in functions_to_overwrite:
                exec(code_string, globals(), exec_dict)
                new_method = exec_dict[name]
                setattr(self.env_cls, name, new_method)