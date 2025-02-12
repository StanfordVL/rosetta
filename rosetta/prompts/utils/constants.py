import pathlib

import rosetta.prompts
from preference_to_reward.maniskill.customized_tasks import *
from preference_to_reward.maniskill.curriculum_learning.customized_tasks import *

# Info 
MANISKILL_ILLEGAL_EDIT_FUNCTIONS = {
    "__init__",
    "_default_sim_config",
    "_default_sensor_configs",
    "_default_human_render_camera_configs",
    "_load_scene",
    "_initialize_episode",
    "_get_obs_extra",
    "compute_normalized_dense_reward",
    "scale_params",
    "reset",
    "task_skill_indices",
    "task_fail"
}

MANISKILL_ACTPRIM_REQUIRED_FUNCTIONS = {
    "skill_reward",
    "evaluate"
}

MANISKILL_CONTCONTROL_REQUIRED_FUNCTIONS = {
    "compute_dense_reward"
}

MANISKILL_ACTPRIM_REMOVAL_FUNCTIONS = {
    "task_skill_indices",
    "_default_sim_config",
    "_default_sensor_configs",
    "_default_human_render_camera_configs",
    "_get_obs_extra",
    "compute_dense_reward",
    "compute_normalized_dense_reward",
    "task_fail",
    "reset",
    "scale_params",
    "_after_reconfigure",
    "_build_bin"
}

MANISKILL_CONTCONTROL_REMOVAL_FUNCTIONS = {
    "_default_sim_config",
    "_default_sensor_configs",
    "_default_human_render_camera_configs",
    "_get_obs_extra",
    "compute_normalized_dense_reward",
    "reset",
    "scale_params"
}

DEFAULT_PARAMS = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_completion_tokens": 4096,
    "top_p": 1,
    "frequency_penalty": 1
}

FUNCS_TO_OVERWRITE = {
    "actprim": ["evaluate", "skill_reward"],
    "contcontrol": ["evaluate", "compute_dense_reward"]
}


# Procedural 
NUM_ERROR_CORR_TRIES = 10


# Content files 
PROMPT_DIR = "prompt_content"

ENV_ID_TO_SIM_CLS = {
    # Main short-horizon continuous control
    "BallAndTarget": BallAndTargetEnv,
    "SphereAndBins": SphereAndBinsEnv,
    
    # Main long-horizon action primitive
    "ObjectsAndBins": ObjectsAndBinsEnv,
    "ObjectsAndDrawer": ObjectsAndDrawerEnv,
    "ThreeCubes": ThreeCubesEnv,
    
    # Baselines + ablations only (short-horizon continuous control)
    "CubeAndPoint": CubeAndPointEnv,
    "CubeAndTarget": CubeAndTargetEnv
}


# Result files 
BACKUP_DIR = pathlib.Path(rosetta.prompts.__file__).parents[1] / "reward_backup"
DEMO_DIR = pathlib.Path(rosetta.prompts.__file__).parents[1] / "demos"
