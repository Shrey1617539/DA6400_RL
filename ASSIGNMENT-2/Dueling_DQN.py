import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    """
    Q-Network for Dueling DQN.
    This network consists of several fully connected layers followed by two heads:
    one for the value function and one for the advantage function.
    """
    def __init__(self, state_size, action_size, num_layers, fc_units_lis, Type):
        super(QNetwork, self).__init__()        
        self.Type = Type
        self.fc_lis = nn.ModuleList()
        self.num_layers = num_layers

        self.fc_lis.append(nn.Linear(state_size, fc_units_lis[0]))
        for i in range(self.num_layers - 1):
            self.fc_lis.append(nn.Linear(fc_units_lis[i],fc_units_lis[i+1]))

        self.adv = nn.Linear(fc_units_lis[-1] , action_size)
        self.value = nn.Linear(fc_units_lis[-1] , 1)

    def forward(self , state):
        x = state
        for i in range(len(self.fc_lis)):
            x = F.relu(self.fc_lis[i](x))
        
        value = self.value(x)
        adv = self.adv(x)

        ## Dueling architecture: combine value and advantage functions
        if self.Type == 'Type 1':
            adv_value = torch.mean(adv, dim=1, keepdim=True)
        
        if self.Type == 'Type 2':
            adv_value = torch.max(adv, dim=1, keepdim=True)[0]

        q_value = value + (adv-adv_value)
        
        return q_value

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device, state_size):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        
        # Pre-allocate fixed numpy arrays (more efficient than deque for large buffers)
        self.state_memory = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, 1), dtype=np.int32)
        self.reward_memory = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.done_memory = np.zeros((buffer_size, 1), dtype=np.float32)
        
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.position
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = done
        
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        # Return batch as numpy arrays
        return (
            self.state_memory[indices],
            self.action_memory[indices],
            self.reward_memory[indices],
            self.next_state_memory[indices],
            self.done_memory[indices]
        )
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
    

class Agent_DDQN:
    """
    Agent for Double DQN.
    This agent uses a Q-Network to learn the optimal action-value function.
    It employs experience replay and a target network to stabilize training.
    """
    def __init__(
            self, state_size, action_size, num_layers, 
            fc_units_lis, policy_param, policy, param_decay, 
            final_param, Type, learning_rate, buffer_size, 
            batch_size, device, discount_rate, update_every
        ):

        self.state_size = state_size
        self.action_size = action_size
        self.fc_units_lis = fc_units_lis
        self.policy_param = policy_param
        self.num_layers = num_layers
        self.policy = policy
        self.param_decay = param_decay
        self.final_param = final_param
        self.Type = Type
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.discount_rate = discount_rate
        self.update_every = update_every

        # Initialize Q-Networks and optimizer
        self.q_local = QNetwork(self.state_size, self.action_size, self.num_layers, self.fc_units_lis, self.Type).to(self.device)
        self.q_target = QNetwork(self.state_size, self.action_size, self.num_layers, self.fc_units_lis, self.Type).to(self.device)
        self.optimizer = optim.Adam(self.q_local.parameters() , lr = self.learning_rate)

        self.memory = ReplayBuffer(self.action_size, buffer_size = self.buffer_size, batch_size = self.batch_size, device = self.device, state_size = self.state_size)

        self.steps = 0

    def get_action(self, action_values):
    # Select an action based on the current policy (epsilon-greedy or softmax).
        if self.policy == 'epsilon_greedy':
            # With probability epsilon, choose random action (exploration)
            if np.random.random() < self.policy_param:
                return np.random.randint(0, self.action_size)
            else:
                # Otherwise choose best action (exploitation)
                return np.argmax(action_values.cpu().numpy())
        
        elif self.policy == 'softmax':
            # Softmax policy: choose actions probabilistically based on their Q-values
            action_lis = action_values.cpu().numpy()[0]
            p_values = np.exp((action_lis-np.max(action_lis))/self.policy_param)
            sum_pi = sum(p_values)
            p_values /= sum_pi

            return np.random.choice(self.action_size , p=p_values)
    
    def step(self,state,action,reward,next_state,done):
        # Store the experience in replay memory
        self.memory.add(state=state,action=action,reward=reward,next_state=next_state,done=done)

        if self.memory.size >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences=experiences,gamma=self.discount_rate)

        self.steps=(self.steps + 1)%self.update_every

        if self.steps  == 0 :
            self.q_target.load_state_dict(self.q_local.state_dict())
    
    def act(self,state):
        # Select an action based on the current policy (epsilon-greedy or softmax).
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()

        with torch.no_grad():
            action_values = self.q_local(state)
        
        self.q_local.train()
        action = self.get_action(action_values=action_values)

        return action

    def learn(self,experiences,gamma):
        # Update the Q-Network using the sampled experiences.
        states , actions , rewards , next_states , dones = experiences

        states  = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        actions_next = self.q_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.q_target(next_states).gather(1, actions_next)
        q_targets = rewards + gamma*q_targets_next*(1-dones)
        q_expected = self.q_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected,q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), max_norm=1)
        self.optimizer.step()

    def decay(self):
        # Decay the policy parameter (epsilon or temperature) over time.
        self.policy_param =  max(self.final_param , self.policy_param*self.param_decay)

def run_experiments(
        env_name, n_episodes, num_expts,learning_rate, 
        discount_factor, policy, param, param_decay,
        final_param, device, fc_units_lis, Type, buffer_size, batch_size, update_every
    ):
    """
    Run multiple experiments on the specified environment using the Dueling DQN agent.
    Each experiment runs for a specified number of episodes and collects rewards.
    The results are averaged across experiments and returned.
    """
    # Set up the environment and agent parameters
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    num_layers = len(fc_units_lis)

    reward_avgs = np.zeros((num_expts, n_episodes))

    # Initialize the agent and run experiments
    for i in range(num_expts):
        print(f'Experiment {i+1}/{num_expts}')
        agent = Agent_DDQN(
            state_size=state_size,
            action_size=action_size,
            num_layers=num_layers,
            fc_units_lis=fc_units_lis,
            policy_param=param,
            policy=policy,
            param_decay=param_decay,
            final_param=final_param,
            Type=Type,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device = device,
            discount_rate=discount_factor,
            update_every=update_every
        )

        # Run the agent in the environment for the specified number of episodes
        for episode in range(n_episodes):
            # Reset the environment and agent for each episode
            state , _ = env.reset()
            done = False
            score = 0

            while not done:
                action = agent.act(state)
                next_state , reward , terminated , truncated , _ = env.step(action)

                done = terminated or truncated

                agent.step(state , action , reward , next_state , done)
                state = next_state
                score += reward
            
            reward_avgs[i][episode] = score
            agent.decay()
    
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