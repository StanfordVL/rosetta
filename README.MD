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
Running examples in the demo:

```bash
python rosetta/run_exp/main_from_csv.py --config_yaml demo/demo.yml
```

After the runningh, the following directory structure will be created:

```
demo/
├── config/    # Configuration files (one folder per data point in the CSV)
├── jsonl/     # CSV data converted to JSONL format
├── result/    # Results (k folders per data point based on number of generated reward functions)
└──...
```


To train policies based on the generated reward functions:

```bash
cd demo/result/[experiment_name]/
bash train_sbatch.sh
```

The training process will generate:
- Video demonstrations of the trained policies
- Sample demonstrations for evaluation
- Trained model weights
