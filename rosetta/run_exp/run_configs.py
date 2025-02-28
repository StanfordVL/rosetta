import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import logging
from feedback_to_reward.run_exp.gen_result_dir import gen_prev_dir_dict, gen_result_dir
from feedback_to_reward.run_exp.gen_sbatch import generate_sbatch
import tyro
from feedback_to_reward.run_exp.utils import BaseConfig
from dataclasses import dataclass, field

@dataclass
class ConfigArgs(BaseConfig):
    config_dirs_path: Optional[Path]=None
    result_dirs_path: Optional[Path]=None
    dry_run: Optional[bool] = True
    num_workers: Optional[int] = 4
    short_prompt_design: Optional[str] = "rosetta_sh"
    long_prompt_design: Optional[str] ="rosetta_lh"
    num_gen: Optional[int] = 1
    chosen_variants: List[int] = field(default_factory=lambda: []) # Param to choose specified index of reward genration when dry_run=False

@dataclass
class ProcessArgs:
    """Arguments for processing a single configuration directory."""
    config_dir_path: Path  # Single config directory path
    prev_dir_dict: Dict
    result_dirs_path: Path  # multiple result directory path
    dry_run: bool
    short_prompt_design: str
    long_prompt_design: str
    num_gen: int = 1
    chosen_variants: List[int] = field(default_factory=lambda: []) 

def process_result_dir(args: ProcessArgs) -> None:
    """Process a single configuration directory."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logging.info(f"Processing configuration directory: {args.config_dir_path}")
    try:
        new_gen_rsts = gen_result_dir(
            src_path=args.config_dir_path, 
            save_dir=args.result_dirs_path,
            result_dict=args.prev_dir_dict, 
            short_prompt_design=args.short_prompt_design,
            long_prompt_design=args.long_prompt_design,
            num_gen=args.num_gen,
            chosen_variants=args.chosen_variants
        )
        for rst in new_gen_rsts:
            sbatch_path = generate_sbatch(rst["folder_path"])
        
            if rst["submittable"] and not args.dry_run:
                subprocess.run(["sbatch", str(sbatch_path)], check=True)
                logging.info(f"Successfully submitted batch job: {sbatch_path}")
            else:
                logging.info(f"Dry run - would submit batch job: {sbatch_path}")
    except Exception as e:
        logging.error(f"Error processing {args.config_dir_path}: {str(e)}")
        raise

def run_configs(
    config_paths: List[Path], 
    result_dirs_path: Path, 
    dry_run: bool, 
    num_workers: int,
    short_prompt_design: Optional[str] = None,
    long_prompt_design: Optional[str] = None,
    num_gen: Optional[int] = 1,
    chosen_variants: Optional[List[int]] = None,
) -> None:
    """Process a list of configuration directories and submit jobs."""
    # Set up default logging configuration if none exists
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    logging.info("Processing configuration directories")
    prev_dir_dict = gen_prev_dir_dict(result_dirs_path)
    if chosen_variants is None:
        chosen_variants = []
    # Create arguments for each config directory
    process_args = [
        ProcessArgs(
            config_dir_path=config_dir_path,
            prev_dir_dict=prev_dir_dict,
            result_dirs_path=result_dirs_path,
            dry_run=dry_run,
            short_prompt_design=short_prompt_design,
            long_prompt_design=long_prompt_design,
            num_gen=num_gen,
            chosen_variants=chosen_variants
        )
        for config_dir_path in config_paths
    ]
    
    # Use ProcessPoolExecutor for parallel processing
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = list(executor.map(process_result_dir, process_args))
            
    except Exception as e:
        logging.error(f"Error in parallel processing: {str(e)}")
        raise

    logging.info("Configuration processing completed successfully")

    
if __name__ == "__main__":
    import os
    config=tyro.cli(ConfigArgs)
    config.load_yaml_config()
    config.config_paths = []
    for dir in os.listdir(config.config_dirs_path):
        config.config_paths.append(os.path.join(config.config_dirs_path, dir))
    run_configs(
        config.config_paths,
        config.result_dirs_path,
        config.dry_run,
        config.num_workers,
        config.short_prompt_design,
        config.long_prompt_design,
        config.num_gen,
        config.chosen_variants,
    )
    
