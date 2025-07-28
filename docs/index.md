# ROSETTA

ROSETTA is a framework that leverages foundation models to interpret natural language preferences, creating multi-stage reward functions that can be implemented through automated code generation.

## Project Overview

ROSETTA bridges the gap between natural language preferences and actionable reward functions for reinforcement learning systems. It uses large language models to understand human-specified preferences and automatically generates code that implements these preferences as reward functions.

## Repository Structure

The ROSETTA repository is organized into several key directories:

```
rosetta/
├── maniskill/      # Environments and training code
├── prompts/        # Prompting pipeline
├── run_exp/        # Running and Managing experiments
└── sb3/            # Some Patch code
```

## Installation Guide

```bash
git clone https://github.com/StanfordVL/rosetta --recursive

conda create -n rosetta python=3.11 -y
conda activate rosetta

cd rosetta
pip install -e .

cd ManiSkill
pip install -e .

cd ../
cd stable-baselines3
pip install -e .
cd ../
```

## Usage Guide

### Running Preference Examples

```bash
python rosetta/run_exp/main_from_csv.py --config_yaml demo/demo.yml
```

This command generates reward functions based on the preferences defined in the demo configuration.
After running the above command, you'll find the following directories:

```
demo/
├── config/    # One config folder per preference
├── jsonl/     # CSV data converted to JSONL format
├── result/    # Result folders per preference based on hyperparameters
└── ...
```

Each result folder contains training scripts and generated reward functions.
To train a policy using the generated reward functions:

```bash
cd demo/result/[experiment_name]/
bash train_sbatch.sh
```

Replace `[experiment_name]` with the name of your specific experiment.
