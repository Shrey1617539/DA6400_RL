[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lDGSs7Pt)

# DA6400_RL_PA1

## Members

1.  Name: Param Jivrajani

    Roll No: PH21B006

2.  Name: Patel Shrey Rakeshkumar

    Roll No: ME21B138

## Project Overview

This project implements reinforcement learning algorithms (Q-Learning and SARSA) to solve the MountainCar-v0 and CartPole-v1 environments from OpenAI Gym. It uses discretization techniques for continuous state spaces and explores different policies (epsilon-greedy and softmax) for action selection.

## File Structure

1. main.py: The main script that runs the experiments.

2. helping_functions.py: Contains helper classes and functions for the RL algorithms.

3. sweep.py: Implements hyperparameter tuning using Weights & Biases (wandb).

4. sweep.yaml: Configuration file for wandb sweeps.

5. requirements.txt: List of required Libraries to run the experiments.

## Key Components

### Main Script (main.py)

- Initializes wandb for experiment tracking (optional).
- Sets up hyperparameters using command-line arguments or wandb config.
- Runs experiments using the run_experiments function.
- Logs results to wandb and plots the results.

### Helper Functions (helping_functions.py)

1. discretization class:
   - Handles state discretization for continuous state spaces.
   - Supports both MountainCar-v0 and CartPole-v1 environments.
2. Agent class:
   - Implements Q-Learning and SARSA algorithms.
   - Supports epsilon-greedy and softmax policies for action selection.
   - Manages Q-table updates and policy parameter decay.
3. run_experiments function:
   - Runs multiple experiments with given parameters.
   - Returns average rewards and steps across experiments.
4. plot_results function:
   - Visualizes experiment results, showing mean rewards and standard deviation.

### Hyperparameter Tuning (sweep.py and sweep.yaml)

- sweep.py: Initializes and runs wandb sweeps for hyperparameter optimization.
- sweep.yaml: Defines the hyperparameter search space and method (Bayesian optimization).

## Usage

1. To run a single experiment:

```
python main.py [arguments]
```

example

```
python main.py -algo "Q-Learning" -bs0 7 -bs1 13 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.007158520376023476 -lr 0.35534122797755796 -mep 200 -pa 1.27093406692732 -pd 0.9906575552866808 -policy "softmax"
```

2. To run hyperparameter sweeps:

```
python sweep.py [arguments]
```

example

```
python sweep.py -we DA6400 -wp PS1
```

## Best Hyper-parameter Result Replication

### CartPole-v1

For SARSA

1.

```
python main.py -algo "SARSA" -bs0 6 -bs1 15 -bs2 22 -bs3 23 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.09800543178221544 -lr 0.22946325984661972 -mep 500 -pa 0.1667503576063203 -pd 0.9827559251881544 -policy "epsilon_greedy"
```

2.

```
python main.py -algo "SARSA" -bs0 8 -bs1 15 -bs2 23 -bs3 25 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.08182626188879606 -lr 0.24764912042029855 -mep 500 -pa 0.4571144619849285 -pd 0.9875113431700852 -policy "epsilon_greedy"
```

3.

```
python main.py -algo "SARSA" -bs0 8 -bs1 9 -bs2 24 -bs3 21 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.06520622831842521 -lr 0.10014948666185836 -mep 500 -pa 0.19906884331367453 -pd 0.999880790079528 -policy "epsilon_greedy"
```

For Q-Learning

1.

```
python main.py -algo "Q-Learning" -bs0 9 -bs1 7 -bs2 10 -bs3 21 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.06207610113920112 -lr 0.46263214388036633 -mep 500 -pa 4.2958259131330045 -pd 0.9893916384579198 -policy "softmax"
```

2.

```
python main.py -algo "Q-Learning" -bs0 14 -bs1 10 -bs2 20 -bs3 13 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.06445193097318799 -lr 0.378920092183182 -mep 500 -pa 4.769136571774245 -pd 0.9979032744055808 -policy "softmax"
```

3.

```
python main.py -algo "Q-Learning" -bs0 12 -bs1 5 -bs2 22 -bs3 21 -env "CartPole-v1" -ep 10000 -exp 5 -fpa 0.05559755322770914 -lr 0.3591346047215045 -mep 500 -pa 3.1323925052084505 -pd 0.998217678770601 -policy "softmax"
```

### MountainCar-v0

For SARSA

1.

```
python main.py -algo "SARSA" -bs0 16 -bs1 17 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.0010247496414152685 -lr 0.153735863132996 -mep 200 -pa 0.7491030659409632 -pd 0.954943311613502 -policy "epsilon_greedy"
```

2.

```
python main.py -algo "SARSA" -bs0 19 -bs1 12 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.0011925083607057503 -lr 0.16292775157888656 -mep 200 -pa 0.9723044941306584 -pd 0.930108764878126 -policy "epsilon_greedy"
```

3.

```
python main.py -algo "SARSA" -bs0 10 -bs1 10 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.0006042359196265721 -lr 0.1645593174927329 -mep 200 -pa 0.8099033852631872 -pd 0.9208323728033362 -policy "epsilon_greedy"
```

For Q-Learning

1.

```
python main.py -algo "Q-Learning" -bs0 6 -bs1 18 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.013910212521735936 -lr 0.33088915907694405 -mep 200 -pa 6.344518406528099 -pd 0.941548018499195 -policy "softmax"
```

2.

```
python main.py -algo "Q-Learning" -bs0 6 -bs1 16 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.002769256799434198 -lr 0.2243513596634139 -mep 200 -pa 2.0001599539089776 -pd 0.9811893194914394 -policy "softmax"
```

3.

```
python main.py -algo "Q-Learning" -bs0 7 -bs1 13 -env "MountainCar-v0" -ep 5000 -exp 5 -fpa 0.007158520376023476 -lr 0.35534122797755796 -mep 200 -pa 1.27093406692732 -pd 0.9906575552866808 -policy "softmax"
```

## Argument Support

### For main.py

| Argument              | Short Flag | Type    | Default                | Description                                                              |
| --------------------- | ---------- | ------- | ---------------------- | ------------------------------------------------------------------------ |
| `--wandb_entity`      | `-we`      | `str`   | `None`                 | Wandb Entity used to track experiments in the Weights & Biases dashboard |
| `--wandb_project`     | `-wp`      | `str`   | `None`                 | Project name used to track experiments in Weights & Biases dashboard     |
| `--learning_rate`     | `-lr`      | `float` | `0.2`                  | Learning rate for training (default: `0.001`)                            |
| `--episodes`          | `-ep`      | `int`   | `3000`                 | Number of episodes (default: `1000`)                                     |
| `--param`             | `-pa`      | `float` | `3.890371023822683`    | Initial parameter value (default: `1.0`)                                 |
| `--param_decay`       | `-pd`      | `float` | `0.9984675934093872`   | Parameter Decay Value (default: `0.99`)                                  |
| `--final_param`       | `-fpa`     | `float` | `0.032306904672786706` | Final parameter value (default: `0.1`)                                   |
| `--experiments`       | `-exp`     | `int`   | `5`                    | Number of experiments (default: `1`)                                     |
| `--env_name`          | `-env`     | `str`   | `"CartPole-v1"`        | Environment name (default: `'CartPole-v1'`)                              |
| `--max_episode_steps` | `-mep`     | `int`   | `200`                  | Maximum steps per episode (default: `200`)                               |
| `--policy`            | `-policy`  | `str`   | `"softmax"`            | Policy type (default: `'greedy'`)                                        |
| `--algo`              | `-algo`    | `str`   | `"Q-Learning"`         | Algorithm type (default: `'Q-learning'`)                                 |
| `--bin_step_0`        | `-bs0`     | `int`   | `17`                   | Number of divisions in 1st state                                         |
| `--bin_step_1`        | `-bs1`     | `int`   | `7`                    | Number of divisions in 2nd state                                         |
| `--bin_step_2`        | `-bs2`     | `int`   | `20`                   | Number of divisions in 3rd state                                         |
| `--bin_step_3`        | `-bs3`     | `int`   | `25`                   | Number of divisions in 4th state                                         |

### For sweep.py

| Argument          | Short Flag | Type  | Default | Description                                                              |
| ----------------- | ---------- | ----- | ------- | ------------------------------------------------------------------------ |
| `--wandb_entity`  | `-we`      | `str` | `None`  | Wandb Entity used to track experiments in the Weights & Biases dashboard |
| `--wandb_project` | `-wp`      | `str` | `None`  | Project name used to track experiments in Weights & Biases dashboard     |
| `--count`         | `-c`       | `int` | `100`   | Maximum number of steps per agent                                        |
