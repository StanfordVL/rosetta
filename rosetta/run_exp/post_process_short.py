import os
import glob
import shutil
import fire
import json
import numpy as np

def post_process_short(src_path: str) -> None:
    """
    Post-process function to extract and rename the latest video and checkpoint.
    
    Args:
        src_path (str): Path to the source directory
    """
    try:
        # Extract folder name
        folder_name = os.path.basename(os.path.normpath(src_path))
        with open(os.path.join(src_path, 'exp_config.json'), 'r') as f:
            exp_config = json.load(f)
        annotator_id = exp_config.get('annotator_id')
        env_id = exp_config.get('env_id')
        uid_reward = exp_config.get('uid_reward')
        video_name = f"{annotator_id}-{env_id}-{uid_reward}"
        # Find latest video in videos_final directory

        # Pick the best demo
        demo_dir = os.path.join(src_path, "demo_dir")
        rollout_infos = sorted(glob.glob(os.path.join(demo_dir, "episode_*", "rollout_info.json")))
        score = []
        for fp in rollout_infos:
            with open(fp, "r") as f:
                rollout_info = json.load(f)
            score.append((rollout_info["is_success"], rollout_info["accumulated_reward"]))
        
        best_rollout_idx = max(range(len(score)), key=lambda i: (score[i][0], score[i][1]))

        # Copy the best rollout to a new directory
        shutil.copytree(rollout_infos[best_rollout_idx].split("rollout_info.json")[0], os.path.join(demo_dir, "best_demo"), dirs_exist_ok=True)
        
        # Copy the best video
        video_dir = os.path.join(src_path, 'demo_dir', 'best_demo')
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        if len(video_files) == 0:
            raise FileNotFoundError(f"No video files found in: {video_dir}")
        video_path = video_files[0]

        # Copy and rename the latest video
        new_video_path = os.path.join(src_path, f"{video_name}.mp4")
        shutil.copy2(video_path, new_video_path)
        
        # Process checkpoint file
        ckpt_path = os.path.join(src_path, 'exp', 'best_model.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
            
        # Copy and rename checkpoint, remove old checkpoint
        new_ckpt_path = os.path.join(src_path, 'best_model.pt')
        shutil.copy2(ckpt_path, new_ckpt_path)
        # os.remove(ckpt_path)
        
        print(f"Post-processing completed successfully:")
        print(f"1. Latest video copied to: {new_video_path}")
        print(f"2. Checkpoint copied to: {new_ckpt_path}")
        
    except Exception as e:
        print(f"Error during post-processing: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(post_process_short)