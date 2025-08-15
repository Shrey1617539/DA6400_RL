import wandb
from helping_functions import discretization, Agent
from helping_functions import run_experiments, plot_results
import argparse
import numpy as np

# Function to get configuration values with priority order: wandb config > command line args > default
def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb if entity and project are provided
    if not(args.wandb_entity is None or args.wandb_project is None):
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    # Set up hyperparameters, using the get_config_value function to determine priority
    discount_factor = 0.99  
    learning_rate = get_config_value(wandb.config, args, 'learning_rate')
    n_episodes = get_config_value(wandb.config, args, 'episodes')
    param = get_config_value(wandb.config, args, 'param')  
    param_decay = get_config_value(wandb.config, args, 'param_decay')  
    final_param = get_config_value(wandb.config, args, 'final_param') 
    num_expts = get_config_value(wandb.config, args, 'experiments') 
    env_name = get_config_value(wandb.config, args, 'env_name')  
    max_episode_steps = get_config_value(wandb.config, args, 'max_episode_steps')  
    policy = get_config_value(wandb.config, args, 'policy') 
    algo = get_config_value(wandb.config, args, 'algo') 

    # Set up discretization bins based on environment
    bin_step_list = []
    if env_name == 'CartPole-v1':
        # CartPole has 4 state dimensions
        for i in range(4):
            bin_step_list.append(get_config_value(wandb.config, args, f'bin_step_{i}'))
    
    elif env_name == 'MountainCar-v0':
        # MountainCar has 2 state dimensions
        for i in range(2):
            bin_step_list.append(get_config_value(wandb.config, args, f'bin_step_{i}'))
    
    # Run the experiments with the configured parameters
    reward_avgs, steps_avgs = run_experiments(
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        bin_step_list=bin_step_list,
        n_episodes=n_episodes,
        num_expts=num_expts,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        policy=policy,
        algo=algo,
        param=param,
        param_decay=param_decay,
        final_param=final_param
    )

    # Calculate average rewards across experiments for each episode
    episode_avg_rewards = np.mean(reward_avgs, axis=0)
    
    optimal_reward = 0
    if env_name == 'CartPole-v1':
        optimal_reward = max_episode_steps
    elif env_name == 'MountainCar-v0':
        optimal_reward = 0

    # Log results to wandb if enabled
    if not(args.wandb_entity is None or args.wandb_project is None):
        # Log reward and regret for each episode
        for rwd in episode_avg_rewards:
            wandb.log({"episode_reward": rwd})
            wandb.log({"episode_regrets": optimal_reward - rwd})  # Assuming 500 is max possible reward
        
        # Log overall metrics
        wandb.log({"cumulative_mean_regret": optimal_reward - np.mean(reward_avgs)})
        wandb.log({"cumulative_mean_reward": np.mean(reward_avgs)})
    
    # Plot the results
    plot_results(reward_avgs)


# Entry point of the script
if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Configure training parameters for the RL agent.")
    
    parser.add_argument('-we', '--wandb_entity', type=str, default=None, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-wp', '--wandb_project', type=str, default=None, help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.2, help="Learning rate for training (default: 0.001)")
    parser.add_argument('-ep', '--episodes', type=int, default=3000, help="Number of episodes (default: 1000)")
    parser.add_argument('-pa', '--param', type=float, default=3.890371023822683, help="Initial parameter value (default: 1.0)")
    parser.add_argument('-pd', '--param_decay', type=float, default=0.9984675934093872, help="Parameter Decay Value (default: 0.99)")
    parser.add_argument('-fpa', '--final_param', type=float, default=0.032306904672786706, help="Final parameter value (default: 0.1)")
    parser.add_argument('-exp', '--experiments', type=int, default=5, help="Number of experiments (default: 1)")
    parser.add_argument('-env', '--env_name', type=str, default="CartPole-v1", help="Environment name (default: 'CartPole-v1')")
    parser.add_argument('-mep', '--max_episode_steps', type=int, default=200, help="Maximum steps per episode (default: 200)")
    parser.add_argument('-policy', '--policy', type=str, default="softmax", help="Policy type (default: 'greedy')")
    parser.add_argument('-algo', '--algo', type=str, default="Q-Learning", help="Algorithm type (default: 'Q-learning')")
    parser.add_argument('-bs0', '--bin_step_0', type=int, default=17, help="number of divisions in 1st state")
    parser.add_argument('-bs1', '--bin_step_1', type=int, default=7, help="number of divisions in 2nd state")
    parser.add_argument('-bs2', '--bin_step_2', type=int, default=20, help="number of divisions in 3rd state")
    parser.add_argument('-bs3', '--bin_step_3', type=int, default=25, help="number of divisions in 4th state")

    # Parse arguments and run main function
    args = parser.parse_args()
    main(args)
