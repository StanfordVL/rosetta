import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI
from rosetta.run_exp.gen_reward import generate_reward
from rosetta.prompts.utils.constants import BACKUP_DIR as backup_path 
from rosetta.run_exp.utils import generate_hash_uid, gen_uid_by_timestamp
from rosetta.run_exp.env_config import ENV_CONFIG
import traceback
import fire
def gen_prev_dir_dict(target_dir: str) -> Dict[str, str]:
    """
    Args:
        target_dir: Root directory to scan for experiment configurations
        
    Returns:
        Dictionary mapping reward UIDs to their abs paths
        
    """
    result_dict = {}
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory {target_dir} does not exist")
    
    dirs = os.listdir(target_dir)
    
    for dir_name in dirs:
        dir_path = os.path.join(target_dir, dir_name)
        config_path = os.path.join(dir_path, "exp_config.json")
        
        if not os.path.isdir(dir_path) or not os.path.exists(config_path):
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                uid_reward = config.get("uid_reward")
                if uid_reward:
                    result_dict[uid_reward] = dir_path
                    
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"Warning: Error processing {config_path}: {str(e)}")
            print(f"Error message: {error_msg}")
            continue
    return result_dict


def gen_result_dir(
    src_path: str,
    save_dir: str,
    result_dict: Dict[str, str],
    num_gen: int = 1,
    short_prompt_design: Optional[str] = "rosetta_sh",
    long_prompt_design: Optional[str] = "rosetta_lh",
    num_retry = 3, # Number of allowed retries for reward generation
    chosen_variants: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    """
    Generate experiment folders with reward functions based on feedback.
    
    This function processes a source experiment folder, generates new reward functions
    based on feedback, and creates new experiment folders with the results. It handles
    the entire pipeline from feedback grounding to reward generation.
    
    Directory structure created:
    save_dir/
    └──{annotator_id}-{env_id}-{prev_uid}-{uid_feedback}-{uid_reward}/
        ├── exp_config.json
        ├── reward.json
        └── reward_gen_rst.json
    
    Args:
        src_path: Path to source experiment folder
        save_dir: Directory where new experiment folders will be created
        result_dict: Mapping of reward UIDs to directory paths
        num_gen: Number of reward functions to generate (default: 1)
    
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source path {src_path} does not exist")
        
    config_path = os.path.join(src_path, 'exp_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    print("handling config: ", exp_config)
    
    try:
        env_id = exp_config['env_id']
        annotator_id = exp_config['annotator_id']
        prev_uid = exp_config['prev_uid']
        uid_feedback = exp_config['uid_feedback']
        feedback = exp_config['feedback']
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")
    
    assert prev_uid in result_dict, f"Previous UID {prev_uid} not found in result_dict, available keys: {list(result_dict.keys())}"
    prev_dir_path = result_dict.get(prev_uid)
    demo_dir = os.path.join(prev_dir_path, 'demo_dir/best_demo')
    # Backward compatibility
    if not Path(demo_dir).exists(): 
        demo_dir = Path(prev_dir_path) / "demo_dir" / "episode_0"
    
    with open(os.path.join(prev_dir_path, 'reward.json'), 'r') as f:
        prev_funcs = json.load(f)
    
    task_description=None
    if os.path.exists(os.path.join(prev_dir_path, 'grounding_rst.json')):
        with open(os.path.join(prev_dir_path, 'grounding_rst.json'), 'r') as f:
            task_description = json.load(f).get("next_description", None)
           
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Ground feedback in demonstrations
    
    # Generate reward functions
    prompt_design=short_prompt_design if ENV_CONFIG[env_id]["task_type"]=="short" else long_prompt_design
    act_space = ENV_CONFIG[env_id]["act_space"]
    reward_results_list = []
    attempt = 0

    # Baseline-specific kwargs
    baseline_kwargs = {}
    if prompt_design=="eureka":
        with open(os.path.join(prev_dir_path, 'exp/results.json'), 'r') as f:
            objective_feedback = json.load(f)
        inner_num_gen = num_gen
        num_gen = 1
        baseline_kwargs = {"objective_feedback": objective_feedback, "inner_num_gen": inner_num_gen}

    while attempt < num_retry + num_gen:
        attempt += 1
        try:
            reward_results = generate_reward(
                client=client,
                exp_id=os.path.basename(src_path),
                human_input=feedback,
                env_id=env_id,
                act_space=act_space,
                task_description=task_description,
                demo_dir=demo_dir,
                prev_funcs=prev_funcs,
                prompt_design=prompt_design,
                simulator=ENV_CONFIG[env_id]["simulator"],
                # Baseline-specific kwargs
                **baseline_kwargs
            )
            reward_results_list.append(reward_results)
        # add error traceback to print detailed error message
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"\nError in reward generation attempt {attempt}:")
            print(f"Error message: {str(error_msg)}")
            continue
        if len(reward_results_list) == num_gen:
            break
    
    # Create experiment folders with results
    
    existing_uids_idx = {}
    existing_uids_round = {}
    folder_paths = []
    for round_idx, rst in enumerate(reward_results_list):
        round_hash = gen_uid_by_timestamp()      
        for idx,reward_func in enumerate(rst["functions"]):
            sorted_func_str = json.dumps(reward_func, sort_keys=True)
            uid_reward = generate_hash_uid(
                str(annotator_id) + str(env_id) + str(prev_uid) + 
                str(uid_feedback) + sorted_func_str
            )
            if uid_reward in existing_uids_idx:
                existing_uids_idx[uid_reward].append(idx)
                existing_uids_round[uid_reward].append(round_idx)
                print(f"Skipping duplicate reward UID: {uid_reward}")
                continue
            
            existing_uids_idx[uid_reward] = [idx]
            existing_uids_round[uid_reward] = [round_idx]
            folder_name = f"{annotator_id}-{env_id}-{prev_uid}-{uid_feedback}-{uid_reward}"
            folder_path = os.path.join(save_dir, folder_name)
            folder_paths.append(folder_path)
            os.makedirs(folder_path, exist_ok=True)
            
            
            exp_config_cur = exp_config.copy()
            exp_config_cur.update({
                "uid_reward": uid_reward,
                "stages": rst["stages"],
                "round_hash": round_hash,
                "backup_folder": rst["backup_folder"]
                
            })
            
           
            file_configs = {
                "exp_config.json": exp_config_cur,
                "reward.json": reward_func,
                "grounding_rst.json": rst["grounding"]
            }
            
            for filename, content in file_configs.items():
                with open(os.path.join(folder_path, filename), 'w') as f:
                    json.dump(content, f, indent=4)
            
    new_gen_rsts = []
    for folder_path in folder_paths: 
        uid_reward = folder_path.split("-")[-1]
        with open(os.path.join(folder_path, "exp_config.json"), "r") as f:
            exp_config = json.load(f)
        cur_reward_variants=existing_uids_idx[uid_reward] # the index of the reward function during one roundß reward generation
        exp_config["reward_variants"] = cur_reward_variants
        exp_config["rounds"] = existing_uids_round[uid_reward]
        with open(os.path.join(folder_path, "exp_config.json"), "w") as f:
            json.dump(exp_config, f, indent=4)
        
        rst={
            "submittable": False,
            "folder_path": folder_path,
        }
        # if cur_reward_variants has intersection with chosen_variants, then submit it
        if not chosen_variants or set(cur_reward_variants).intersection(set(chosen_variants)):
            rst["submittable"] = True
        else:
            print(f"Skipping reward UID: {uid_reward}, reward variants: {cur_reward_variants}, due to chosen_variants: {chosen_variants}")
        new_gen_rsts.append(rst)

    return new_gen_rsts 
            


if __name__ == "__main__":
    fire.Fire(gen_result_dir)