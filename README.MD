# Rosetta

## Overview
ROSETTA is a framework that leverages foundation models to interpret natural language preferences, creating multi-stage reward functions that can be implemented through automated code generation.


## Structure

```
rosetta/
├── maniskill/      # Environments and training code
├── prompts/        # Prompting pipeline
├── run_exp/        # Running and Managing experiments
└── sb3/            # Some Patch code
```

## Getting Started

### Installation

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

### Quickstart
We provide the preference examples:

```bash
python rosetta/run_exp/main_from_csv.py --config_yaml demo/demo.yml
```

The reward functions will be generated in the result folders:

```
demo/
├── config/    # one config folder per preference
├── jsonl/     # CSV data converted to JSONL format
├── result/    # n result folders per perference based on hyper-param, each folder contains training scripts and reward functions
└──...
```

You can train the policy by running:

```bash
cd demo/result/[experiment_name]/
bash train_sbatch.sh
```
