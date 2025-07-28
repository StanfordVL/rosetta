

import os
import json
from pathlib import Path
import fire
from dataclasses import dataclass, field
from rosetta.run_exp.utils import BaseConfig
from rosetta.run_exp.query_history_videos import batch_build_history_and_option_folder
from dataclasses import dataclass
from pathlib import Path
from typing import List
import tyro
# batch_build_history_and_option_folder(config_dirs_path: str,result_dirs_path:str, num_ancestors:int, save_path='./test')

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import fire

@dataclass
class config(BaseConfig):
    input_paths: List[Path] = field(default_factory=list)  # Use default_factory for mutable defaults
    output_dir: Path = Path("./test")
    num_ancestors: int = 1
    form_links: List[str] = field(default_factory=list)
    remove_mapping: bool=True
    
# please strcture the folder as follows for the input paths:
        # --path
        # ----configs_dir
        # ----results_dir
    
def main(config):
    # Get all round_{num_ancestors} directories for each input path
    round_dirs = {}
    for path in config.input_paths:
        round_dir = path / f"round{config.num_ancestors}"
        if not round_dir.exists():
            # Create round directory if it doesn't exist
            batch_build_history_and_option_folder(
                str(path / "configs"),
                str(path / "results"),
                config.num_ancestors,
                str(round_dir)
            )
        round_dirs[path] = round_dir

    # Collect all config directories from each path
    config_dirs_by_path = {}
    for path, round_dir in round_dirs.items():
        config_dirs = [d for d in round_dir.iterdir() if d.is_dir()]
        config_dirs_by_path[path] = {d.name: d for d in config_dirs}

    # Find common config directories across all paths
    common_config_names = set.intersection(
        *[set(dirs.keys()) for dirs in config_dirs_by_path.values()]
    )

    # Process each common config directory
    for config_name in common_config_names:
        # Create output directory for this config
        output_config_dir = config.output_dir / config_name
        output_config_dir.mkdir(parents=True, exist_ok=True)

        # Create options directory
        options_dir = output_config_dir / "options-to-choose-from"
        options_dir.mkdir(exist_ok=True)

        # Create mapping dictionary for this config
        mapping = {}
        
        # Keep track of next available number for video renaming
        next_video_num = 1
        video_mapping = {}  # maps original video names to new numbered names

        # Process options-to-choose-from for each path
        for path in config.input_paths:
            exp_name = os.path.basename(str(path))
            source_options_dir = config_dirs_by_path[path][config_name] / "options-to-choose-from"
            
            if not source_options_dir.exists():
                continue

            path_mapping = []
            
            # Process each video in the options directory
            for video_file in source_options_dir.glob("*.mp4"):
                original_name = video_file.name
                
                # Check if we've already assigned a number to this video name
                if original_name not in video_mapping:
                    new_name = f"{next_video_num}.mp4"
                    video_mapping[original_name] = new_name
                    next_video_num += 1
                else:
                    new_name = video_mapping[original_name]
                
                # Copy the video file with the new name
                import shutil
                shutil.copy2(video_file, options_dir / new_name)
                
                # Add to path mapping
                path_mapping.append({new_name:original_name})
            
            mapping[exp_name] = path_mapping

        # Save mapping to JSON file
        with open(output_config_dir / "mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)

        # Copy other files (excluding options-to-choose-from directory)
        # We can take files from any path since they should be the same
        source_dir = config_dirs_by_path[config.input_paths[0]][config_name]
        for item in source_dir.iterdir():
            if item.name != "options-to-choose-from" and item.name != "mapping.json":
                if item.is_dir():
                    shutil.copytree(item, output_config_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, output_config_dir / item.name)
                    
    for idx, form_link in enumerate(config.form_links):
        # Add evaluation form links to the original output directory
        add_eval_form_links(config.output_dir, form_link, f"link_{idx}.txt")

    if config.remove_mapping:
        # Create a new directory for the no-mapping version
        no_mapping_dir = Path(str(config.output_dir) + "_no_mapping")
        
        # Copy the entire directory structure
        import shutil
        if no_mapping_dir.exists():
            shutil.rmtree(no_mapping_dir)
        shutil.copytree(config.output_dir, no_mapping_dir)
        
        # Remove mapping files from the new directory
        remove_mapping_files(no_mapping_dir)
        



def add_eval_form_links(base_dir, form_link: str,form_link_name='link.txt'):
    """
    Add evaluation form link to each config directory.
    
    Args:
        base_dir (Path): The base directory containing config directories
        form_link (str): The evaluation form link to be added
    """
    # Iterate through all immediate subdirectories (config dirs)
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    for config_dir in base_dir.iterdir():
        if config_dir.is_dir():
            # Create link.txt in each config directory
            link_file = os.path.join(config_dir, form_link_name)
            with open(link_file, "w") as f:
                f.write(form_link)

def remove_mapping_files(base_dir: Path):
    """
    Remove mapping.json from each config directory.
    
    Args:
        base_dir (Path): The base directory containing config directories
    """
    # Iterate through all immediate subdirectories (config dirs)
    for config_dir in base_dir.iterdir():
        if config_dir.is_dir():
            # Look for mapping.json and remove if it exists
            mapping_file = config_dir / "mapping.json"
            if mapping_file.exists():
                mapping_file.unlink()

# Example usage:
# add_eval_form_links(Path("./output_dir"), "https://your-form-link.com")
# remove_mapping_files(Path("./output_dir"))

if __name__ == "__main__":
    tyro.cli(
        main,
        config,
        description="Combine videos from multiple paths into a single directory structure.",
    )
    
