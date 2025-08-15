import wandb
from Dueling_DQN import run_experiments as Dueling_DQN
from MC_REINFORCE import run_experiments as MC_REINFORCE
from Dueling_DQN import plot_results
import argparse
import numpy as np
import torch

# Function to get configuration values with priority order: wandb config > command line args > default
def get_config_value(config, args, key, default=None):
    if config is None:
        return getattr(args, key, default)
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb if entity and project are provided
    wandb_config = None
    use_wandb = not(args.wandb_entity is None or args.wandb_project is None)
    if use_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb_config = wandb.config

    # Set up hyperparameters, using the get_config_value function to determine priority
    algo = get_config_value(wandb_config, args, 'algo')
    env_name = get_config_value(wandb_config, args, 'env_name', args.env_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run the experiments with the configured parameters
    if algo == 'DDQN':
        # Initialize Dueling DQN agent with the provided parameters
        reward_avgs = Dueling_DQN(
            env_name=get_config_value(wandb_config, args, 'env_name', args.env_name),
            n_episodes=get_config_value(wandb_config, args, 'episodes', args.episodes),
            num_expts=get_config_value(wandb_config, args, 'experiments', args.experiments),
            learning_rate=get_config_value(wandb_config, args, 'learning_rate', args.learning_rate),
            discount_factor=get_config_value(wandb_config, args, 'discount_factor', 0.99),
            policy=get_config_value(wandb_config, args, 'policy', args.policy),
            param=get_config_value(wandb_config, args, 'param', args.param),
            param_decay=get_config_value(wandb_config, args, 'param_decay', args.param_decay),
            final_param=get_config_value(wandb_config, args, 'final_param', args.final_param),
            device=device,
            fc_units_lis=get_config_value(wandb_config, args, 'fc_units_lis'),
            Type=get_config_value(wandb_config, args, 'Type'),
            buffer_size=get_config_value(wandb_config, args, 'buffer_size', 1e6),
            batch_size=get_config_value(wandb_config, args, 'batch_size', 64),
            update_every=get_config_value(wandb_config, args, 'update_every', 4)
        )

    elif algo == 'MC_REINFORCE':
        # Initialize Monte Carlo REINFORCE agent with the provided parameters
        reward_avgs = MC_REINFORCE(
            env_name=get_config_value(wandb_config, args, 'env_name', args.env_name),
            n_episodes=get_config_value(wandb_config, args, 'episodes', args.episodes),
            num_expts=get_config_value(wandb_config, args, 'experiments', args.experiments),
            discount_factor=get_config_value(wandb_config, args, 'discount_factor', 0.99),
            fc_units_lis=get_config_value(wandb_config, args, 'fc_units_lis'),
            baseline=get_config_value(wandb_config, args, 'baseline', True),
            lr1=get_config_value(wandb_config, args, 'learning_rate1', args.learning_rate1),
            lr2=get_config_value(wandb_config, args, 'learning_rate2', args.learning_rate2),
            device=device,
        )

    # Calculate average rewards across experiments for each episode
    episode_avg_rewards = np.mean(reward_avgs, axis=0)
    
    optimal_reward = 0
    if env_name == 'CartPole-v1':
        optimal_reward = 500
    elif env_name == 'Acrobot-v1':
        optimal_reward = -100

    # Log results to wandb if enabled
    if use_wandb:
        # Log reward and regret for each episode
        for rwd in episode_avg_rewards:
            wandb.log({"episode_reward": rwd})
            wandb.log({"episode_regrets": optimal_reward - rwd})
        
        # Log overall metrics
        wandb.log({"cumulative_mean_regret": optimal_reward - np.mean(reward_avgs)})
        wandb.log({"cumulative_mean_reward": np.mean(reward_avgs)})

    plot_results(reward_avgs=reward_avgs, rolling_length=5)

# Entry point of the script
if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Configure training parameters for the RL agent.")
    
    # Basic configuration
    parser.add_argument('-we', '--wandb_entity', type=str, default=None, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-wp', '--wandb_project', type=str, default=None, help='Wandb Project name for tracking experiments')

    # Algorithm selection
    parser.add_argument('-a', '--algo', type=str, default='DDQN', choices=['DDQN', 'MC_REINFORCE'], help='RL algorithm to use (DDQN or MC_REINFORCE)')

    # Environment settings
    parser.add_argument('-e', '--env_name', type=str, default='CartPole-v1', help='Name of the Gym environment')
    parser.add_argument('-ep', '--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('-ex', '--experiments', type=int, default=3, help='Number of experiments to run and average')

    # Learning parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.000209, help='Learning rate for the optimizer')
    parser.add_argument('-lr1', '--learning_rate1', type=float, default=1e-4, help='Learning rate for the policy network')
    parser.add_argument('-lr2', '--learning_rate2', type=float, default=1e-4, help='Learning rate for the value network')
    parser.add_argument('-df', '--discount_factor', type=float, default=0.99, help='Discount factor for future rewards')

    # Policy parameters
    parser.add_argument('-p', '--policy', type=str, default='softmax', choices=['epsilon_greedy', 'softmax'], help='Policy type for action selection')
    parser.add_argument('-pa', '--param', type=float, default=9.788, help='Initial exploration parameter (epsilon for epsilon-greedy)')
    parser.add_argument('-pd', '--param_decay', type=float, default=0.9535, help='Decay rate for exploration parameter')
    parser.add_argument('-fp', '--final_param', type=float, default=0.411, help='Final exploration parameter value')

    # Network architecture
    parser.add_argument('-fc', '--fc_units_lis', nargs="*", type=int, default=[256,128], help='List of fully connected layer units for neural network')
    parser.add_argument('-t', '--Type', type=str, default='Type 2', choices=['Type 1', 'Type 2'], help='Type of DQN architecture to use')
    parser.add_argument('-bs', '--buffer_size', type=int, default=1000000, help='Replay buffer size for experience replay')
    parser.add_argument('-bsize', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-ue', '--update_every', type=int, default=100, help='How often to update the network')

    # MC REINFORCE specific
    parser.add_argument('-b', '--baseline', type=bool, default=True, help='Whether to use baseline in MC REINFORCE algorithm')

    # Parse arguments and run main function
    args = parser.parse_args()
    main(args)
