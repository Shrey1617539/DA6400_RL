# DA6400_RL_PA2

## Members

1. Name: Param Jivrajani

   Roll No: PH21B006

2. Name: Patel Shrey Rakeshkumar

   Roll No: ME21B138

## Overview

This project implements reinforcement learning experiments using two algorithms: **Dueling Double DQN (DDQN)** and **Monte Carlo REINFORCE**. The code provides the following features:

- **Training and evaluation of RL agents:**
  - **DDQN:** Uses a dueling architecture with separate value and advantage streams.
  - **MC REINFORCE:** Implements a policy gradient method with an optional baseline.
- **Experiment Sweeps:**  
  Hyperparameter tuning and sweeps are integrated via Weights & Biases (wandb) using a YAML configuration file.
- **Result Visualization:**  
  Each algorithm has built-in routines to plot the average rewards and moving averages over episodes.

---

## File Structure

```
├── Dueling_DQN.py # Dueling DQN agent and supporting functions
├── MC_REINFORCE.py # Monte Carlo REINFORCE agent and supporting functions
├── main.py # Main script for training and evaluation
├── sweep.py # Script for launching wandb hyperparameter sweeps
├── sweep.yaml # Sweep configuration for wandb
└── requirements.txt # Project dependencies
```

---

## Key Components

### `Dueling_DQN.py`

- Defines the Dueling DQN architecture with value and advantage streams.
- `Agent_DDQN` handles action selection, experience replay, and training.
- Includes:
  - `ReplayBuffer`
  - `run_experiments()` for training across multiple seeds
  - `plot_results()` for visualization

### `MC_REINFORCE.py`

- Implements policy-gradient REINFORCE algorithm.
- `Agent_MC` optionally uses a baseline (value function).
- Includes:
  - `make_nn()` for custom FC network
  - `run_experiments()` for evaluation
  - `plot_results()` for visualization

### `main.py`

- Entry point for training either DDQN or MC_REINFORCE agents.
- Integrates with wandb for experiment logging.
- Allows customization via CLI or wandb sweeps.

### `sweep.yaml`

- Defines a Bayesian optimization sweep with configurable ranges:
  - Learning rate, architecture, policy parameters, and more
- Supports both DDQN and MC REINFORCE

### `sweep.py`

- Loads `sweep.yaml`, registers the sweep with wandb
- Accepts wandb entity/project and number of runs via CLI

### `requirements.txt`

- Lists required Python packages:
  - `torch`, `gym`, `matplotlib`, `wandb`, etc.

---

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Experiments

```
python main.py [arguments]
```

examples

```
python main.py -a DDQN -e CartPole-v1 -ep 500 -ex 3 -fc_units_lis 256, 128
```

With wandb logging

```
python main.py -a MC_REINFORCE -e Acrobot-v1 -we your_entity -wp your_project
```

### 3. Launch Hyperparameter Sweep

```
python sweep.py --wandb_entity your_entity --wandb_project your_project --count 100

```

## Best Hyper-parameter Result Replication

### For Monte Carlo Reinforce

1. ## CartPole-v1

   1. ### Without Baseline

   ```
   python main.py --algo MC_REINFORCE --baseline False --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.00011928405159438484 --learning_rate2 0.013078516239705306

   python main.py --algo MC_REINFORCE --baseline False --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --learning_rate1 0.0004949887448459288 --learning_rate2 0.00007299356085063911

   python main.py --algo MC_REINFORCE --baseline False --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.0005494727066085914 --learning_rate2 0.00010656239467423532
   ```

   2. ### With Baseline

   ```
   python main.py --algo MC_REINFORCE --baseline True --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.0010922846648572717 --learning_rate2 0.048845316213569155

   python main.py --algo MC_REINFORCE --baseline True --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.0006481339596904447 --learning_rate2 0.013397493168627456

   python main.py --algo MC_REINFORCE --baseline True --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.0007697293408386931 --learning_rate2 0.018911401885139657
   ```

2. ## Acrobot-v1

   1. ### Without Baseline

   ```
   python main.py --algo MC_REINFORCE --baseline False --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.0010920066584220618 --learning_rate2 0.016778919175737077

   python main.py --algo MC_REINFORCE --baseline False --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --learning_rate1 0.0012085134994885322 --learning_rate2 0.023133880998021827

   python main.py --algo MC_REINFORCE --baseline False --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --learning_rate1 0.000909278973016768 --learning_rate2 0.06198542963273099
   ```

   2. ### With Baseline

   ```
   python main.py --algo MC_REINFORCE --baseline True --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --learning_rate1 0.0010904764123757349 --learning_rate2 0.03207102086603413

   python main.py --algo MC_REINFORCE --baseline True --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --learning_rate1 0.0006119438580542343 --learning_rate2 0.027421882925871725

   python main.py --algo MC_REINFORCE --baseline True --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --learning_rate1 0.001477406701503032 --learning_rate2 0.026090050715996165
   ```

### For Dueling DQN

1. ## CartPole-v1

   1. ### Type - 1

   ```
   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.17341350854988802 --learning_rate 0.0002518139228491362 --param 0.40477401702030696 --param_decay 0.990132538178658 --policy epsilon_greedy --Type "Type 1" --update_every 200

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.13769495610118287 --learning_rate 0.00019195304526787064 --param 0.27264709737024184 --param_decay 0.9904984904175774 --policy epsilon_greedy --Type "Type 1" --update_every 200

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.15901449024433825 --learning_rate 0.0005782517582333243 --param 0.4007950304830677 --param_decay 0.9903839820441988 --policy epsilon_greedy --Type "Type 1" --update_every 200
   ```

   2. ### Type - 2

   ```
   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.41134136418272593 --learning_rate 0.0002093332064484207 --param 9.78826237858749 --param_decay 0.95350306087557 --policy softmax --Type "Type 2" --update_every 100

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.3790945427338607 --learning_rate 0.0002670870052632571 --param 9.11302144281692 --param_decay 0.9591331078167452 --policy softmax --Type "Type 2" --update_every 100

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "CartPole-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.7522578292998199 --learning_rate 0.0003575078850419479 --param 11.698684529941527 --param_decay 0.9632446455073328 --policy softmax --Type "Type 2" --update_every 200
   ```

2. ## Acrobot-v1

   1. ### Type - 1

   ```
   python main.py --algo DDQN --batch_size 256 --buffer_size 500000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 --final_param 0.03205952754596827 --learning_rate 0.00047280045993327576 --param 0.5345502220787033 --param_decay 0.9775672119265651 --policy epsilon_greedy --Type "Type 1" --update_every 100

   python main.py --algo DDQN --batch_size 256 --buffer_size 1000000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 32 --final_param 0.14002045525928727 --learning_rate 0.0003858230230519132 --param 0.05434522179396521 --param_decay 0.9599403699709168 --policy epsilon_greedy --Type "Type 1" --update_every 100

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 32 --final_param 0.14148624309837177 --learning_rate 0.0008008131778176055 --param 3.9903717228054703 --param_decay 0.9500899574972892 --policy softmax --Type "Type 1" --update_every 100
   ```

   2. ### Type - 2

   ```
   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 32 --final_param 0.12627791796254975 --learning_rate 0.0006334472398332822 --param 0.4730780931924319 --param_decay 0.962897849252265 --policy softmax --Type "Type 2" --update_every 100

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 256 128 --final_param 0.1238231143907386 --learning_rate 0.0006933360938151919 --param 0.49532338835177314 --param_decay 0.9623448406976975 --policy softmax --Type "Type 2" --update_every 100

   python main.py --algo DDQN --batch_size 128 --buffer_size 1000000 --env_name "Acrobot-v1" --episodes 1000 --experiments 5 --fc_units_lis 128 64 32 --final_param 0.12300565153713126 --learning_rate 0.000971215473330972 --param 0.32910743611453835 --param_decay 0.9641028283077684 --policy softmax --Type "Type 2" --update_every 100
   ```

## Argument Support

### For main.py

| Argument            | Short Flag | Type    | Default       | Description                                                              |
| ------------------- | ---------- | ------- | ------------- | ------------------------------------------------------------------------ |
| `--wandb_entity`    | `-we`      | `str`   | `None`        | Wandb Entity used to track experiments in the Weights & Biases dashboard |
| `--wandb_project`   | `-wp`      | `str`   | `None`        | Wandb Project name for tracking experiments                              |
| `--algo`            | `-a`       | `str`   | `DDQN`        | RL algorithm to use                                                      |
| `--env_name`        | `-e`       | `str`   | `CartPole-v1` | Name of the Gym environment                                              |
| `--episodes`        | `-ep`      | `int`   | `500`         | Number of episodes to train                                              |
| `--experiments`     | `-ex`      | `int`   | `3`           | Number of experiments to average                                         |
| `--learning_rate`   | `-lr`      | `float` | `0.000209`    | Learning rate for DDQN optimizer                                         |
| `--learning_rate1`  | `-lr1`     | `float` | `1e-4`        | Learning rate for policy network (MC_REINFORCE)                          |
| `--learning_rate2`  | `-lr2`     | `float` | `1e-4`        | Learning rate for value network (MC_REINFORCE)                           |
| `--discount_factor` | `-df`      | `float` | `0.99`        | Discount factor for future rewards                                       |
| `--policy`          | `-p`       | `str`   | `softmax`     | Action selection policy (`softmax` or `epsilon_greedy`)                  |
| `--param`           | `-pa`      | `float` | `9.788`       | Initial exploration parameter (epsilon or softmax temp)                  |
| `--param_decay`     | `-pd`      | `float` | `0.9535`      | Decay rate for the exploration parameter                                 |
| `--final_param`     | `-fp`      | `float` | `0.411`       | Final exploration parameter value                                        |
| `--fc_units_lis`    | `-fc`      | `int[]` | `[256, 128]`  | List of fully connected layer units separated by comma                   |
| `--Type`            | `-t`       | `str`   | `Type 2`      | Dueling DQN architecture type (`Type 1` or `Type 2`)                     |
| `--buffer_size`     | `-bs`      | `int`   | `1000000`     | Replay buffer size                                                       |
| `--batch_size`      | `-bsize`   | `int`   | `64`          | Batch size for training                                                  |
| `--update_every`    | `-ue`      | `int`   | `100`         | Frequency of target network updates                                      |
| `--baseline`        | `-b`       | `bool`  | `True`        | Whether to use baseline in MC REINFORCE                                  |

### For sweep.py

| Argument          | Short Flag | Type  | Default | Description                                         |
| ----------------- | ---------- | ----- | ------- | --------------------------------------------------- |
| `--wandb_entity`  | `-we`      | `str` | `None`  | Wandb Entity used to track sweep experiments        |
| `--wandb_project` | `-wp`      | `str` | `None`  | Wandb Project name for sweep tracking               |
| `--count`         | `-c`       | `int` | `100`   | Maximum number of sweeps/agents to run in the sweep |
