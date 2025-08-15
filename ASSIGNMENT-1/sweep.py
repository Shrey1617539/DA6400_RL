import wandb
import yaml
import argparse

# Initialize sweep with data from .yaml file
with open("sweep.yaml", "r") as file:
    sweep_configuration = yaml.safe_load(file)
wandb.require("core")

# Getting arguments for entity and project name
parser = argparse.ArgumentParser(
    description="Training script that return model weights"
)
parser.add_argument(
    '-we',
    '--wandb_entity',
    type=str,
    default=None,
    help='Wandb Entity used to track experiments in the Weights & Biases dashboard'
)
parser.add_argument(
    '-wp',
    '--wandb_project',
    type=str,
    default=None,
    help='Project name used to track experiments in Weights & Biases dashboard'
)
parser.add_argument(
    '-c',
    '--count',
    type = int,
    default=100,
    help = 'Maximum number of seep per agent'
)

args = parser.parse_args()

sweep_id = wandb.sweep(
    sweep_configuration,
    entity=args.wandb_entity,
    project=args.wandb_project
)

# Initilizing agent
wandb.agent(sweep_id, count=args.count)
