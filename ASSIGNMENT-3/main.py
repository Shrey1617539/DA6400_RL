import wandb
import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from Options_Agents import Option_Agent, Option_Agent_alt, plot_value_policy, plot_options_value_policy

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
    option_choice = get_config_value(wandb_config, args, 'option_choice')
    env = gym.make('Taxi-v3', render_mode='ansi')


    num_exp = get_config_value(wandb_config, args, 'experiments')
    num_episodes = get_config_value(wandb_config, args, 'episodes', args.episodes)
    reward_avgs = np.zeros((num_exp, num_episodes))
    completed_task_lis = np.zeros((num_exp, num_episodes))

    for exp in tqdm(range(num_exp)):
        if option_choice == 1:
            action_map = {
                0: 'south', 1: 'north', 2: 'east', 3: 'west',
                4: 'pick', 5: 'drop', 6: 'reachR', 7: 'reachG',
                8: 'reachY', 9: 'reachB'
            }
            agent = Option_Agent(
                env = env,
                q_table = np.zeros((env.observation_space.n, 10)),
                state_shape=env.observation_space.n,
                action_shape=env.action_space.n,
                option_shape=4,
                action_map = action_map,
                policy=get_config_value(wandb_config, args, 'policy'),
                param=get_config_value(wandb_config, args, 'param'),
                final_param=get_config_value(wandb_config, args, 'final_param'),
                param_decay=get_config_value(wandb_config, args, 'param_decay'),
                discount_factor=get_config_value(wandb_config, args, 'discount_factor', 0.99),
                learning_rate=get_config_value(wandb_config, args, 'learning_rate', args.learning_rate),
                algo=algo
            )
        elif option_choice == 2:
            action_map = {
                0: 'south', 1: 'north', 2: 'east', 3: 'west',
                4: 'pick', 5: 'drop', 6: 'gopickup', 7: 'godropoff'
            }
            agent = Option_Agent_alt(
                env = env,
                q_table = np.zeros((env.observation_space.n, 8)),
                state_shape=env.observation_space.n,
                action_shape=env.action_space.n,
                option_shape=2,
                action_map = action_map,
                policy=get_config_value(wandb_config, args, 'policy'),
                param=get_config_value(wandb_config, args, 'param'),
                final_param=get_config_value(wandb_config, args, 'final_param'),
                param_decay=get_config_value(wandb_config, args, 'param_decay'),
                discount_factor=get_config_value(wandb_config, args, 'discount_factor', 0.99),
                learning_rate=get_config_value(wandb_config, args, 'learning_rate', args.learning_rate),
                algo=algo
            )

        # Reset the environment for each experiment
        state, info = env.reset()
        done = False
        episode_reward = 0
        completed_task = 0

        for episode in tqdm(range(num_episodes)):
            # Reset the environment for each episode
            state, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Decode the state and get the action from the agent
                action = agent.get_action(state_idx=state)

                if action <= 5:
                    # Perform the action in the environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    agent.update(state_idx=state, action=action, reward=reward,
                                 terminated=terminated, next_state_idx=next_state)
                    episode_reward += reward
                
                else:
                    # Perform the option in the environment
                    total_reward, undecayed_reward, steps, next_state, terminated, truncated = agent.Option(state, action)
                    if agent.algo == 'smdp':
                        future_q = 0 if terminated or truncated else np.max(agent.Q[next_state])
                        agent.Q[state][action] += agent.learning_rate * (total_reward + (agent.discount_factor ** steps) * future_q - agent.Q[state][action])
                    episode_reward += undecayed_reward

                state = next_state
                done = terminated or truncated

            final_state_decode = agent.decode(state=state)
            if final_state_decode[-2] == final_state_decode[-1]:
                completed_task += 1
            
            reward_avgs[exp][episode] = episode_reward
            completed_task_lis[exp][episode] = completed_task
            agent.decay()
    # np.save(f'results/{agent.algo}_{option_choice}_reward_avgs.npy', reward_avgs)
    # Calculate average rewards across experiments for each episode
    episode_avg_rewards = np.mean(reward_avgs, axis=0)
    optimal_reward = 0

    # plot_value_policy(agent, save_path = 'plots')
    # plot_options_value_policy(agent, save_path='plots', option_choice=option_choice)
    plot_value_policy(agent, save_path = 'plots', option_choice=option_choice)

    # Log results to wandb if enabled
    if use_wandb:
        # Log reward and regret for each episode
        for i, rwd in enumerate(episode_avg_rewards):
            wandb.log({"episode_reward": rwd})
            wandb.log({"episode_regrets": optimal_reward - rwd})
            wandb.log({"episode_completed_tasks": np.mean(completed_task_lis, axis=0)[i]})
        
        # Log overall metrics
        wandb.log({"cumulative_mean_regret": optimal_reward - np.mean(reward_avgs)})
        wandb.log({"cumulative_mean_reward": np.mean(reward_avgs)})

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Run Option-based Reinforcement Learning")
    parser.add_argument('--wandb_entity', type=str, default='DA6400_PA1_param_shrey', help='WandB entity name')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('--learning_rate', type=float, default=0.4355000413860653, help='Learning rate')
    parser.add_argument('--discount_factor', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--policy', type=str, default="softmax", help='Policy to use')
    parser.add_argument('--param', type=float, default=18.37855462526911, help='Initial parameter for exploration')
    parser.add_argument('--final_param', type=float, default=0.05464339270596094, help='Final parameter for exploration decay')
    parser.add_argument('--param_decay', type=float, default=0.9308627588783698, help='Decay rate for exploration parameter')
    parser.add_argument('--option_choice', type=int, default=1, help='Option choice (1 or 2)')
    parser.add_argument('--algo', type=str, default="intra_option", help='Algorithm to use (DDQN or MC_REINFORCE)')

    args = parser.parse_args()
    
    main(args)