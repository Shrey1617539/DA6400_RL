import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

class make_nn(nn.Module):
    """
    A neural network class for policy and value networks.
    It consists of multiple fully connected layers with ReLU activations.
    The last layer outputs the action probabilities or value estimates.
    """
    def __init__(
            self, state_size, action_size, 
            num_layers, fc_units_lis, device, softmax_use
        ):
        super(make_nn, self).__init__()        
        self.fc_lis = nn.ModuleList()
        self.num_layers = num_layers
        self.fc_lis.append(nn.Linear(state_size, fc_units_lis[0]))

        for i in range(self.num_layers - 1):
            self.fc_lis.append(nn.Linear(fc_units_lis[i], fc_units_lis[i+1]))
        self.fc_lis.append(nn.Linear(fc_units_lis[-1], action_size))

        self.softmax_use = softmax_use
        if self.softmax_use:
            self.softmax = nn.Softmax(dim=-1)

        self.to(device)

    def forward(self , state):
        x = state

        for i in range(len(self.fc_lis)-1):
            x = F.relu(self.fc_lis[i](x))

        x = self.fc_lis[-1](x)

        if self.softmax_use:
            x = self.softmax(x)
        return x

class Agent_MC:
    """
    A Monte Carlo REINFORCE agent for policy gradient methods.
    It uses a neural network to approximate the policy and optionally a value function.
    The agent learns from episodes of experience, updating the policy based on the returns.
    """
    def __init__(
            self, state_size, action_size, num_layers, 
            fc_units_lis, baseline, lr1, lr2, device
        ):
        self.state_size = state_size
        self.action_size = action_size
        self.fc_units_lis = fc_units_lis
        self.num_layers = num_layers
        self.baseline = baseline
        self.lr1 = lr1
        self.device = device

        # Initialize policy network
        self.policy_network = make_nn(
            state_size=self.state_size,
            action_size=self.action_size,
            num_layers=self.num_layers,
            fc_units_lis=self.fc_units_lis,
            device=self.device,
            softmax_use=True
        )
        self.optimizer1 = optim.Adam(self.policy_network.parameters(), lr=self.lr1)

        if baseline:
            # Initialize value network
            self.lr2 = lr2
            self.value_network = make_nn(
                state_size=self.state_size,
                action_size=1,
                num_layers=self.num_layers,
                fc_units_lis=self.fc_units_lis,
                device=self.device,
                softmax_use=False
            )
            self.optimizer2 = optim.Adam(self.value_network.parameters(), lr=self.lr2)

    # Get action from the policy network based on the current state.        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = torch.unsqueeze(state, 0)
        prob = self.policy_network.forward(state)
        prob = torch.squeeze(prob, 0)
        action = prob.multinomial(num_samples=1)
        action = action.item()
        return action
    
    # Get action probabilities from the policy network for a given state.
    def get_prob(self, state, action):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        prob = self.policy_network.forward(state)
        prob = torch.squeeze(prob, 0)
        return prob[action]
    
    # Get the value estimate from the value network for a given state.
    def get_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        value = self.value_network.forward(state)
        value = torch.squeeze(value, 0)
        return value

    # Learn from the collected experience (states, actions, rewards)
    def learn(self, state_lis, action_lis, reward_lis, gamma):
        states = torch.tensor(np.array(state_lis), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(action_lis), dtype=torch.long).to(self.device)  # Changed to long
        rewards = torch.tensor(np.array(reward_lis), dtype=torch.float32).to(self.device)

        returns = []
        G = 0.0
        # Calculate returns using the rewards and discount factor
        for i, reward in enumerate(reversed(reward_lis)):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        if self.baseline:
            # Value network update with TD(0)
            current_values = self.value_network(states).squeeze()
            
            # Calculate next state values
            next_states = [state_lis[i+1] if i < len(state_lis)-1 else None 
                        for i in range(len(state_lis))]
            
            # Create mask and next values
            non_terminal_mask = torch.tensor(
                [ns is not None for ns in next_states], 
                dtype=torch.bool
            ).to(self.device)
            
            next_values = torch.zeros(len(next_states)).to(self.device)
            if non_terminal_mask.any():
                non_terminal_next_states = torch.tensor(
                    np.array([ns for ns in next_states if ns is not None]),
                    dtype=torch.float32
                ).to(self.device)
                
                with torch.no_grad():
                    next_values[non_terminal_mask] = self.value_network(
                        non_terminal_next_states
                    ).squeeze()
            
            # Calculate TD targets
            td_targets = rewards + gamma * next_values
            
            # Value network update
            value_loss = F.mse_loss(current_values, td_targets.detach())
            self.optimizer2.zero_grad()
            value_loss.backward()
            self.optimizer2.step()
            current_values = self.value_network(states).squeeze()
            advantages = (returns - current_values.detach())
        else:
            advantages = returns

        # Policy network update
        action_probs = self.policy_network(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = (-torch.log(selected_probs + 1e-8) * advantages).mean()
        
        self.optimizer1.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer1.step()


def run_experiments(
        env_name, n_episodes, num_expts, discount_factor, 
        fc_units_lis, baseline, lr1, lr2, device
    ):
    """
    Run multiple experiments with the specified parameters and return average rewards.
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    reward_avgs = np.zeros((num_expts, n_episodes))
    num_layers = len(fc_units_lis)

    # Initialize the agent and run experiments
    for i in range(num_expts):
        agent = Agent_MC(
            state_size=state_size,
            action_size=action_size,
            num_layers=num_layers,
            fc_units_lis=fc_units_lis,
            baseline=baseline,
            lr1=lr1,
            lr2=lr2,
            device=device
        )

        # Run episodes for the current experiment
        for episode in range(n_episodes):
            state_lis = []
            action_lis = []
            reward_lis = []
            state, _ = env.reset()

            score = 0
            done = False

            while not done:
                action = agent.get_action(state)

                state_lis.append(state)
                action_lis.append(action)
                nxt_state, reward, termination, truncation, _ = env.step(action)
                
                done = termination or truncation
    
                reward_lis.append(reward)
                score += reward

                if done:
                    break

                state = nxt_state

            # Update the agent with the collected experience
            agent.learn(state_lis, action_lis, reward_lis, discount_factor)

            reward_avgs[i][episode] = score
    
    env.close()
    return reward_avgs

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

    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    reward_moving_avgs = np.zeros((reward_avgs.shape[0], reward_avgs.shape[1]-rolling_length+1))
    for i in range(reward_avgs.shape[0]):
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
    plt.title('Moving '+title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.plot(np.arange((mean_values_moving.shape[0])), mean_values_moving, label='Mean', color='blue', linewidth=2)
    plt.fill_between(np.arange(len(mean_values_moving)), 
                    mean_values_moving -np.sqrt(variance_values_moving), 
                    mean_values_moving + np.sqrt(variance_values_moving), 
                    color='lightblue', alpha=0.5, label='± Standard Deviation')
    plt.show()