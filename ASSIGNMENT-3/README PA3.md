
# DA6400_RL_PA3

## Members

1.  Name: Param Jivrajani  
    Roll No: PH21B006
    
    **Work**: Code for SMDP and Results in the report

3.  Name: Patel Shrey Rakeshkumar  
    Roll No: ME21B138
    
    **Work**: Code for Intra-Option and Plot generation in report

---

## Project Overview

This project applies option-based reinforcement learning to the Taxi-v3 environment using two approaches:
- **Intra-Option Q-Learning**
- **Semi-Markov Decision Process (SMDP)**

The code supports two predefined **option structures**, leverages action abstraction for efficient learning, and uses either **epsilon-greedy** or **softmax** policies. Experiment logging and hyperparameter optimization are integrated with **Weights & Biases (wandb)**.

---

## File Structure

1. `main.py`: Main script to run experiments and evaluate agents.
2. `Options_Agents.py`: Contains `Option_Agent` and `Option_Agent_alt` classes, implementing the learning logic and value/policy plotting.
3. `sweep.py`: Automates hyperparameter sweeps using wandb.
4. `sweep.yaml`: Configuration file defining the sweep method and search space.

---

## Key Components

### Main Script (`main.py`)
- Loads Taxi-v3 environment from Gym.
- Configures agent hyperparameters via command-line or wandb.
- Supports two types of option frameworks (`Option_Agent`, `Option_Agent_alt`).
- Logs results and visualizations.
- Tracks rewards, task completions, and value function convergence.

### Agent Definitions (`Options_Agents.py`)
- `Option_Agent`:
  - Implements 4 discrete options: `reachR`, `reachG`, `reachY`, `reachB`.
- `Option_Agent_alt`:
  - Implements 2 options: `gopickup` and `godropoff`.
- Both agents:
  - Support intra-option and SMDP update styles.
  - Allow flexible exploration strategies via epsilon-greedy or softmax.
  - Include Q-learning updates and option-specific decay.
  - Provide rich visualizations of learned value functions and policies.

### Hyperparameter Sweep Configuration (`sweep.yaml`)
- Uses Bayesian optimization to search over:
  - Learning rate
  - Exploration parameter (`param`)
  - Decay rate (`param_decay`)
  - Final parameter (`final_param`)
  - Policy type
- Set to minimize `cumulative_mean_regret` for optimal performance.

### Sweep Runner (`sweep.py`)
- Loads sweep configuration from `sweep.yaml`.
- Initializes a sweep on wandb with provided entity and project.
- Spawns a wandb agent to run multiple experiments.
- Supports limiting the number of sweep runs via the `--count` argument.

Example usage:

```bash
python sweep.py --wandb_entity "your_entity" --wandb_project "PA3" --count 50
```

---

## Usage

### Run a Single Experiment

```bash
python main.py --algo "intra_option" --episodes 10000 --experiments 5 --learning_rate 0.435 \
--discount_factor 0.99 --policy "softmax" --param 18.379 --param_decay 0.931 --final_param 0.054 \
--option_choice 1
```

### Run a Sweep

```bash
python sweep.py --wandb_entity "your_entity" --wandb_project "PA3"
```

---

## Example Arguments (`main.py`)

| Argument            | Short Flag | Type    | Default     | Description                                   |
|---------------------|------------|---------|-------------|-----------------------------------------------|
| `--wandb_entity`    | `-we`      | `str`   | None        | Wandb entity name                             |
| `--wandb_project`   | `-wp`      | `str`   | None        | Wandb project name                            |
| `--episodes`        | `-ep`      | `int`   | 10000       | Number of episodes                            |
| `--experiments`     | `-exp`     | `int`   | 1           | Number of runs per configuration              |
| `--learning_rate`   | `-lr`      | `float` | 0.435       | Q-learning learning rate                      |
| `--discount_factor` | `-df`      | `float` | 0.99        | Discount factor                               |
| `--policy`          | `-policy`  | `str`   | "softmax"   | Policy type (`epsilon_greedy` or `softmax`)   |
| `--param`           | `-pa`      | `float` | 18.379      | Initial exploration parameter                 |
| `--param_decay`     | `-pd`      | `float` | 0.931       | Decay rate for exploration parameter          |
| `--final_param`     | `-fpa`     | `float` | 0.054       | Final exploration parameter                   |
| `--option_choice`   | `-oc`      | `int`   | 1           | Option structure: 1 (4 options) or 2 (2 options) |
| `--algo`            | `-algo`    | `str`   | "intra_option" | Algorithm: `intra_option` or `smdp`          |

---

## Notes

- All experiments are run on **Taxi-v3** from OpenAI Gym.
- The reward, regret, and completion metrics are automatically logged if wandb is enabled.
- Learned policies and value functions are saved in the `plots/` directory.
