import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class discretization:
    """
    Class to handle state discretization for continuous state spaces.
    Converts continuous state values into discrete bins for tabular RL methods.
    """
    def __init__(self, env_name, bin_step_list):
        self.env_name = env_name
        if env_name == 'MountainCar-v0':
            # Define bins for position and velocity for MountainCar
            self.POS_BINS = np.linspace(-1.2, 0.6, num=bin_step_list[0], endpoint=False)
            self.VEL_BINS = np.linspace(-0.07, 0.07, num=bin_step_list[1], endpoint=False)
            self.NUM_ACTIONS = 3
        
        elif env_name == 'CartPole-v1':
            # Define bins for position, velocity, angle, and angular velocity for CartPole
            self.POS_BINS = np.linspace(-4.8, 4.8, num=bin_step_list[0], endpoint=False)
            self.VEL_BINS = np.linspace(-4, 4, num=bin_step_list[1], endpoint=False)
            self.ANG_BINS = np.linspace(-0.5, 0.5, num=bin_step_list[2], endpoint=False)
            self.AV_BINS = np.linspace(-4, 4, num=bin_step_list[3], endpoint=False)

    def discretize_state(self, state):
        """
        Convert a continuous state into a discrete state representation (bin indices).
        Uses np.digitize to find which bin each state component belongs to.
        """
        if self.env_name == 'MountainCar-v0':
            # Discretize position and velocity for MountainCar
            pos_idx = np.clip(np.digitize(state[0], self.POS_BINS) - 1, 0, len(self.POS_BINS)-1)
            vel_idx = np.clip(np.digitize(state[1], self.VEL_BINS) - 1, 0, len(self.VEL_BINS)-1)
            return pos_idx, vel_idx

        elif self.env_name == 'CartPole-v1':
            # Discretize all four state variables for CartPole
            pos_idx = np.clip(np.digitize(state[0], self.POS_BINS) - 1, 0, len(self.POS_BINS)-1)
            vel_idx = np.clip(np.digitize(state[1], self.VEL_BINS) - 1, 0, len(self.VEL_BINS)-1)
            ang_idx = np.clip(np.digitize(state[2], self.ANG_BINS) - 1, 0, len(self.ANG_BINS)-1)
            av_idx = np.clip(np.digitize(state[3], self.AV_BINS) - 1, 0, len(self.AV_BINS)-1)
            return pos_idx, vel_idx, ang_idx, av_idx

class Agent:
    """
    RL agent implementing Q-Learning and SARSA algorithms with different policies.
    """
    def __init__(self, learning_rate, discount_factor, policy, algo, policy_param, param_decay, final_param, num_actions, bin_step_list):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.policy = policy
        self.policy_param = policy_param 
        self.param_decay = param_decay 
        self.final_param = final_param 
        self.algo = algo
        self.num_actions = num_actions 
        self.bin_step_list = bin_step_list 
        # Initialize Q-table with zeros (state dimensions + action dimension)
        self.Q = np.zeros(self.bin_step_list + [self.num_actions])
    
    def get_action(self, state_idx):
        """
        Select an action based on the current policy (epsilon-greedy or softmax).
        """
        if self.policy == 'epsilon_greedy':
            # With probability epsilon, choose random action (exploration)
            if np.random.random() < self.policy_param:
                return np.random.randint(0, self.num_actions)
            else:
                # Otherwise choose best action (exploitation)
                return np.argmax(self.Q[state_idx])
        
        elif self.policy == 'softmax':
            # Softmax policy: choose actions probabilistically based on their Q-values
            q_values = self.Q[state_idx] / self.policy_param
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(self.num_actions, p=probabilities)
        
        return 0
    
    def update(self, state_idx, action, reward, terminated, next_state_idx):
        """
        Update Q-values using either Q-Learning or SARSA algorithm.
        """
        if self.algo == 'Q-Learning':
            # Q-Learning: Use max Q-value of next state (off-policy)
            q_next = 0 if terminated else np.max(self.Q[next_state_idx])
            td_error = reward + self.discount_factor * q_next - self.Q[state_idx][action]
            self.Q[state_idx][action] += self.learning_rate * td_error
        
        elif self.algo == 'SARSA':
            # SARSA: Use Q-value of next state-action pair (on-policy)
            next_action = self.get_action(next_state_idx)
            q_next = 0 if terminated else self.Q[next_state_idx][next_action]
            td_error = reward + self.discount_factor * q_next - self.Q[state_idx][action]
            self.Q[state_idx][action] += self.learning_rate * td_error
    
    def decay_param(self):
        """
        Decay the policy parameter (epsilon or temperature) over time.
        """
        self.policy_param = max(self.final_param, self.policy_param*self.param_decay)

def run_experiments(env_name, max_episode_steps, bin_step_list, n_episodes, num_expts, learning_rate, discount_factor, policy, algo, param, param_decay, final_param):
    """
    Run multiple experiments with the same parameters to get statistical results.
    """
    env = gym.make(env_name)
    env._max_episode_steps = max_episode_steps 
    dis = discretization(env_name, bin_step_list=bin_step_list) 
    reward_avgs = np.zeros((num_expts, n_episodes))
    steps_avgs = np.zeros((num_expts, n_episodes))
    
    for i in range(num_expts):
        # Create a new agent for each experiment
        agent = Agent(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            policy=policy,
            algo=algo,
            policy_param=param,
            param_decay=param_decay,
            final_param=final_param,
            num_actions=env.action_space.n,
            bin_step_list=bin_step_list
        )
        
        # Run n_episodes for this experiment
        for episode in tqdm(range(n_episodes), desc=f"Experiment {i+1}/{num_expts}"):
            obs, _ = env.reset()
            done = False
            score = 0 
            steps = 0
            
            # Episode loop
            while not done:
                state_idx = dis.discretize_state(obs) 
                action = agent.get_action(state_idx)  
                next_obs, reward, terminated, truncated, _ = env.step(action) 
                next_state_idx = dis.discretize_state(next_obs)  
                
                # Update agent's Q-values
                agent.update(state_idx, action, reward, terminated, next_state_idx)
                
                score += reward
                steps += 1
                done = terminated or truncated  
                obs = next_obs  
            
            # Decay exploration parameter after each episode
            agent.decay_param()  

            # Record total reward and steps taken
            reward_avgs[i, episode] = score  
            steps_avgs[i, episode] = steps  

    env.close()
    return reward_avgs, steps_avgs

def plot_results(reward_avgs, rolling_length = 100):
    """
    Plot the results of experiments, showing mean and variance of rewards.
    """
    title = 'Average Reward vs Episodes'
    mean_values = np.mean(reward_avgs, axis=0) 
    variance_values = np.var(reward_avgs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange((mean_values.shape[0])), mean_values, label='Mean', color='blue', linewidth=2)

    # Plot standard deviation as shaded area
    plt.fill_between(np.arange(len(mean_values)), 
                    mean_values -np.sqrt(variance_values), 
                    mean_values + np.sqrt(variance_values), 
                    color='lightblue', alpha=0.5, label='± Standard Deviation')

    plt.title('Moving '+title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    reward_moving_avgs = np.zeros((reward_avgs.shape[0], reward_avgs.shape[1]-rolling_length+1))
    for i in range(5):
        # Convolve with window of size rolling_length to get moving average
        reward_moving_avgs[i] = np.convolve(
            reward_avgs[i],
            np.ones(rolling_length) / rolling_length,
            mode="valid"
        )
    
    # Calculate mean and variance of moving averages
    mean_values_moving = np.mean(reward_moving_avgs, axis=0)
    variance_values_moving = np.var(reward_moving_avgs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange((mean_values_moving.shape[0])), mean_values_moving, label='Mean', color='blue', linewidth=2)

    plt.fill_between(np.arange(len(mean_values_moving)), 
                    mean_values_moving -np.sqrt(variance_values_moving), 
                    mean_values_moving + np.sqrt(variance_values_moving), 
                    color='lightblue', alpha=0.5, label='± Standard Deviation')
