import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import tyro
from typing import Annotated
import os
from rosetta.run_exp.get_csv import csv_to_jsonl
from rosetta.run_exp.gen_config_dirs import gen_config_dirs
from rosetta.run_exp.run_configs import run_configs
from rosetta.run_exp.utils import BaseConfig
from typing import List


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for experiment pipeline."""

    source_file_path: Annotated[Optional[Path], "Path to load source file"] = './test/test.csv'

    jsonl_output_path: Annotated[Optional[Path], "Path to save JSONL output"] = './test/output.jsonl'
    config_dirs_path: Annotated[Optional[Path], "Path to configuration directories"] = './test/configs'
    result_dirs_path: Annotated[Optional[Path], "Path to results directories"] = './test/results'
    dry_run: Annotated[Optional[bool], "Run without submitting jobs"] = False
    num_workers: Annotated[Optional[int], "Number of worker processes (default: CPU count)"] = 4
    short_prompt_design: Annotated[Optional[str], "Short prompt design"] = "rosetta_sh"
    long_prompt_design: Annotated[Optional[str], "Long prompt design"] = "rosetta_lh"
    num_gen: Annotated[Optional[int], "Number of reward to generate"] = 1
    chosen_variants: List[int] = field(default_factory=lambda: [5,6]) # Param to choose specified index of reward genration when dry_run=False

    config_yaml: Annotated[Optional[Path], "Path to YAML configuration file"] = None



    def validate_paths(self) -> None:
        """
        Validate that all required directories exist or can be created and convert relative paths to absolute.

        Raises:
            ValueError: If any path is invalid or cannot be created
        """
        paths = {
            "jsonl_output": self.jsonl_output_path,
            "config_dirs": self.config_dirs_path,
            "result_dirs": self.result_dirs_path
        }

        for name, path in paths.items():
            path_str = str(path)
            abs_path = os.path.abspath(path_str)
            paths[name] = Path(abs_path)
            setattr(self, f"{name}_path", Path(abs_path))

        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> None:
        if not os.path.exists(self.source_file_path):
            raise ValueError(f"Local file {self.source_file_path} does not exist.")

def run_experiment(config: ExperimentConfig) -> None:
    """Run the experiment pipeline from .csv or .json to batch job submission."""
    logging.info("Starting experiment pipeline")
    if config.source_file_path.endswith('.csv'):
        logging.info(f"Fetching feedback from local file {config.source_file_path}")
        csv_to_jsonl(
            config.csv_path,
            config.jsonl_output_path
        )
    else:
        # copy from local
        logging.info(f"Fetching feedback from local file {config.source_file_path}")
        assert config.source_file_path.endswith('.jsonl')
        if not (Path(os.path.abspath(config.source_file_path)) == config.jsonl_output_path):
            shutil.copyfile(config.source_file_path, config.jsonl_output_path)


    logging.info("Generating configuration directories")
    newly_added_config_dir_paths = gen_config_dirs(
        config.jsonl_output_path,
        config.config_dirs_path
    )
    print(newly_added_config_dir_paths)

    if not newly_added_config_dir_paths:
        logging.warning("No new configurations were generated")
        return

    run_configs(
        config_paths=newly_added_config_dir_paths,
        result_dirs_path=config.result_dirs_path,
        dry_run=config.dry_run,
        num_workers=config.num_workers,
        short_prompt_design=config.short_prompt_design,
        long_prompt_design=config.long_prompt_design,
        num_gen=config.num_gen,
        chosen_variants=config.chosen_variants
    )

    logging.info("Experiment pipeline completed successfully")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        config = tyro.cli(ExperimentConfig)
        config.load_yaml_config()
        config.validate_paths()
        config.validate_config()
        print(config)
        run_experiment(config)
    except Exception as e:
        logging.error(str(e))
        raise