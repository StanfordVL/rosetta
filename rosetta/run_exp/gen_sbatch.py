import os
import json
from typing import Dict, Any
import fire
from feedback_to_reward.run_exp.env_config import ENV_CONFIG
from pathlib import Path

from feedback_to_reward.run_exp.system_config import ACCOUNT, PARTITION, EMAIL

# Common SLURM header template shared between long and short tasks
slurm_header_template = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output={working_dir}/slurm-%j.out
#SBATCH --error={working_dir}/slurm-%j.err
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user={email}
ulimit -c 102400

cd {working_dir}
"""

# Training command templates
train_cmd_long_template = """python -m feedback_to_reward.maniskill.long_horizon_learning.sb3_skill_maple_stage \\
    --save-path {save_path} \\
    --env_id {env_id} \\
    --reward_json_path {reward_json} \\
    --max_episode_steps {max_episode_steps}
"""

train_cmd_short_template = """python -m feedback_to_reward.maniskill.short_horizon_learning.maniskill_ppo \\
    --exp_name {exp_path} \\
    --reward_json_path {reward_json} \\
    --default_config {default_config}
"""

# Post-processing command template shared between long and short tasks
post_process_template_short = """python -m feedback_to_reward.run_exp.record_demo \\
    --env_id {env_id} \\
    --checkpoint {checkpoint_path} \\
    --out_dir {demo_dir} \\
    --task_type {task_type} \\
    --num_demo 10 \\
    --sample_freq 5 \\
    --reward_json_path {reward_json} \\
    --cuda

python -m feedback_to_reward.run_exp.post_process_{task_type} {working_dir}
"""

post_process_template_long = """
python -m feedback_to_reward.run_exp.record_demo \\
    --env_id {env_id} \\
    --checkpoint {checkpoint_path} \\
    --out_dir {demo_dir} \\
    --task_type {task_type} \\
    --max_episode_steps {max_episode_steps} \\
    --num_demo 10 \\
    --reward_json_path {reward_json} \\
    --cuda

python -m feedback_to_reward.run_exp.post_process_{task_type} {working_dir}
"""


def generate_sbatch_short(src_path: str) -> str:
    """Generate SLURM batch files for short tasks."""
    with open(os.path.join(src_path, 'exp_config.json'), 'r') as f:
        exp_config = json.load(f)
    
    env_id = exp_config['env_id']
    env_config = ENV_CONFIG[env_id]
    folder_name = os.path.basename(os.path.normpath(src_path))
    
    # Save default training config
    default_config = os.path.join(src_path, 'default_training_config.json')
    with open(default_config, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Common parameters
    params = {
        'cpus_per_task': 4,  # Specific to short tasks
        'job_name': folder_name,
        'job_mem': env_config['job_mem'],
        'working_dir': src_path,
        'env_id': env_id,
        'reward_json': 'reward.json',
        'default_config': 'default_training_config.json',
        'exp_path': f"{src_path}/exp",
        'checkpoint_path': f"{src_path}/exp/best_model.pt",
        'demo_dir': f"{src_path}/demo_dir",
        'task_type': 'short',
        'account': ACCOUNT,
        'partition': PARTITION,
        'exp_id': Path(src_path).stem,
        'email': EMAIL
    }
    
    # Generate complete batch script
    sbatch_content = (
        slurm_header_template.format(**params) +
        train_cmd_short_template.format(**params) +
        post_process_template_short.format(**params)
    )
    
    output_path = os.path.join(src_path, 'train_sbatch.sh')
    with open(output_path, 'w') as f:
        f.write(sbatch_content)
    
    return output_path

def generate_sbatch_long(src_path: str) -> str:
    """Generate SLURM batch files for long tasks."""
    with open(os.path.join(src_path, 'exp_config.json'), 'r') as f:
        exp_config = json.load(f)
    
    env_id = exp_config['env_id']
    num_stages = int(exp_config['stages'])
    env_config = ENV_CONFIG[env_id]
    folder_name = os.path.basename(os.path.normpath(src_path))
    
    # Save default training config
    default_config = os.path.join(src_path, 'default_training_config.json')
    with open(default_config, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Common parameters
    params = {
        'cpus_per_task': 18,  # Specific to long tasks
        'job_name': folder_name,
        'job_mem': env_config['job_mem'],
        'working_dir': src_path,
        'env_id': env_id,
        'reward_json': 'reward.json',
        'default_config': 'default_training_config.json',
        'demo_dir': f"{src_path}/demo_dir",
        'checkpoint_path': f"{src_path}/exp/eval/best_model/best_model.zip",
        'task_type': 'long',
        'max_episode_steps': num_stages,
        'save_path': "exp",
        'account': ACCOUNT,
        'partition': PARTITION,
        'exp_id': Path(src_path).stem,
        'email': EMAIL

    }

    # Generate batch script with header
    sbatch_content = slurm_header_template.format(**params)
    sbatch_content += train_cmd_long_template.format(**params)
    
    # Add post-processing
    sbatch_content += post_process_template_long.format(**params)
    
    output_path = os.path.join(src_path, 'train_sbatch.sh')
    with open(output_path, 'w') as f:
        f.write(sbatch_content)
    
    return output_path

def generate_sbatch(src_path: str) -> str:
    """Generate appropriate SLURM batch files based on task type."""
    with open(os.path.join(src_path, 'exp_config.json'), 'r') as f:
        exp_config = json.load(f)
        
    env_id = exp_config['env_id']
    task_type = ENV_CONFIG[env_id]['task_type']
    
    if task_type == 'short':
        return generate_sbatch_short(src_path)
    elif task_type == 'long':
        return generate_sbatch_long(src_path)
    else:
        raise ValueError("Invalid task_type. Must be 'short' or 'long'")

if __name__ == "__main__":
    fire.Fire(generate_sbatch)