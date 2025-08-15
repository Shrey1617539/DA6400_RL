import numpy as np

class Option_Agent:
    """
    Option Agent for the Taxi environment.
    This agent implements the options framework for reinforcement learning.
    It can learn both intra-option and inter-option policies.
    """
    def __init__(
            self, env, q_table, state_shape, action_shape, option_shape, action_map,
            policy, param, final_param, param_decay, discount_factor, learning_rate, algo
        ):
        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.option_shape = option_shape
        self.policy = policy
        self.param = param
        self.final_param = final_param
        self.param_decay = param_decay
        self.action_map = action_map
        self.Q = q_table
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.algo = algo
        self.Q_opt_lis = [np.zeros((5, 5, action_shape)) for _ in range(option_shape)]
        self.param_opt_lis = [param for _ in range(option_shape)]
        
    def decode(self, state):
        """
        Decode the state from the environment into its components.
        The state is represented as a single integer, and we decode it into
        taxi_row, taxi_column, passenger_location, and destination.
        """
        taxi_row = state // 100
        taxi_column = (state // 20) % 5
        passenger_location = (state % 20) // 4
        destination = (state % 20) % 4
        return (taxi_row, taxi_column, passenger_location, destination)

    def get_action(self, state_idx):
        """
        Get the action to take based on the current policy.
        The action can be either a normal action or an option.
        """
        actions = self.action_shape + self.option_shape

        # If the policy is epsilon-greedy, choose a random action with probability epsilon
        # Otherwise, choose the action with the highest Q-value
        if self.policy == 'epsilon_greedy':
            if np.random.random() < self.param:
                return np.random.randint(0, actions)
            else:
                return np.argmax(self.Q[state_idx][:actions])
        
        # If the policy is softmax, calculate the probabilities of each action
        # based on their Q-values and sample an action from the distribution
        
        elif self.policy == 'softmax':
            q_values = self.Q[state_idx][:actions] / self.param
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(actions, p=probabilities)
        
        return 0
    
    def get_opt_action(self, state_idx, opt_id):
        """
        Get the action to take based on the current policy for a specific option.
        The action can be either a normal action or an option.
        """
        actions = self.action_shape

        # If the policy is epsilon-greedy, choose a random action with probability epsilon
        # Otherwise, choose the action with the highest Q-value
        if self.policy == 'epsilon_greedy':
            if np.random.random() < self.param_opt_lis[opt_id]:
                return np.random.randint(0, actions)
            else:
                return np.argmax(self.Q_opt_lis[opt_id][state_idx][:actions])
        
        # If the policy is softmax, calculate the probabilities of each action
        # based on their Q-values and sample an action from the distribution
        elif self.policy == 'softmax':
            q_values = self.Q_opt_lis[opt_id][state_idx][:actions] / self.param_opt_lis[opt_id]
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(actions, p=probabilities)
        return 0

    # Check if the option is done based on the state
    def is_red(self, state_decode):
        return (state_decode[0] == 0 and state_decode[1] == 0)

    def is_green(self, state_decode):
        return (state_decode[0] == 0 and state_decode[1] == 4)
    
    def is_yellow(self, state_decode):
        return (state_decode[0] == 4 and state_decode[1] == 0)
    
    def is_blue(self, state_decode):
        return (state_decode[0] == 4 and state_decode[1] == 3)
    
    def Option(self, state_idx, option_id):
        """
        Execute the option in the environment.
        The option can be either a pickup or dropoff option.
        The option is executed until it is done or the episode terminates.
        """
        optdone = False
        state = state_idx
        tot_reward = 0
        undecay_reward = 0
        steps = 0
        terminated, truncated = False, False
        opt_idx = option_id - 6
        max_option_steps = 100

        # Loop until the option is done or the episode terminates
        while not optdone and not terminated and not truncated and steps < max_option_steps:
            state_decode = self.decode(state)
            taxi_row, taxi_col = state_decode[0], state_decode[1]

            if option_id == 6 and self.is_red(state_decode):
                optdone = True
            if option_id == 7 and self.is_green(state_decode):
                optdone = True
            if option_id == 8 and self.is_yellow(state_decode):
                optdone = True
            if option_id == 9 and self.is_blue(state_decode):
                optdone = True

            if optdone:
                
                if (state_decode[-2] == option_id - 6) or (state_decode[-1] == option_id - 6):
                    if state_decode[-2] == option_id - 6:
                        next_state, reward, terminated, truncated, _ = self.env.step(action=4)
                        optaction = 4
                    elif state_decode[-1] == option_id - 6:
                        next_state, reward, terminated, truncated, _ = self.env.step(action=5)
                        optaction = 5

                    
                    next_state_decode = self.decode(state=next_state)
                    next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]

                    # Update the Q-values based on the algorithm
                    # If the algorithm is intra-option, update the Q-values for both the option and the action
                    if self.algo == 'intra_option':
                        self.update(state_idx=state, action=optaction, reward=reward,
                                    terminated=terminated, next_state_idx=next_state)
                        
                        # Update the Q-values for all options that took the same action
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                        self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                        for opt_id in range(self.option_shape):
                            
                            if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                                self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                                reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                                self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                    # If the algorithm is smdp, update the Q-values for the option only
                    elif self.algo == 'smdp':
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                    
                    undecay_reward += reward
                    tot_reward += reward * (self.discount_factor ** steps)
                    state = next_state
                    steps += 1

                else:
                    optaction = self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_idx)
                    next_state, reward, terminated, truncated, _ = self.env.step(action=optaction)
                    next_state_decode = self.decode(state=next_state)
                    next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]
                    # Update the Q-values based on the algorithm
                    # If the algorithm is intra-option, update the Q-values for both the option and the action
                    if self.algo == 'intra_option':
                        self.update(state_idx=state, action=optaction, reward=reward,
                                    terminated=terminated, next_state_idx=next_state)
                        
                        # Update the Q-values for all options that took the same action
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                        self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                        for opt_id in range(self.option_shape):
                            
                            if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                                self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                                reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                                self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                    # If the algorithm is smdp, update the Q-values for the option only
                    elif self.algo == 'smdp':
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))

                    undecay_reward += reward
                    tot_reward += reward * (self.discount_factor ** steps)
                    steps += 1
                    state = next_state

                
            else:
                # Get the action for the option
                # Execute the action in the environment and get the next state and reward
                optaction = self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_idx)
                next_state, reward, terminated, truncated, _ = self.env.step(action=optaction)
                next_state_decode = self.decode(state=next_state)
                next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]

                # Update the Q-values based on the algorithm
                # If the algorithm is intra-option, update the Q-values for both the option and the action
                if self.algo == 'intra_option':
                    self.update(state_idx=state, action=optaction, reward=reward,
                                terminated=terminated, next_state_idx=next_state)
                    
                    # Update the Q-values for all options that took the same action
                    self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                    reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                    self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                    for opt_id in range(self.option_shape):
                        
                        if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                            self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                            reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                            self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                # If the algorithm is smdp, update the Q-values for the option only
                elif self.algo == 'smdp':
                    self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                    reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))

                undecay_reward += reward
                tot_reward += reward * (self.discount_factor ** steps)
                steps += 1
                state = next_state

        # Decay the option parameter after the option is done
        self.decay_opt(opt_id=opt_idx)
        return tot_reward, undecay_reward, steps, state, terminated, truncated

    def update_opt(self, opt_id, state_idx, action, reward, terminated, next_state_idx):
        """
        Update the Q-values for the option based on the reward and next state.
        The Q-values are updated using the TD error.
        """
        q_next = 0 if terminated else np.max(self.Q_opt_lis[opt_id][next_state_idx])
        td_error = reward + self.discount_factor * q_next - self.Q_opt_lis[opt_id][state_idx][action]
        self.Q_opt_lis[opt_id][state_idx][action] += self.learning_rate * td_error

    def update(self, state_idx, action, reward, terminated, next_state_idx):
        """
        Update the Q-values for the action based on the reward and next state.
        The Q-values are updated using the TD error.
        """
        q_next = 0 if terminated else np.max(self.Q[next_state_idx])
        td_error = reward + self.discount_factor * q_next - self.Q[state_idx][action]
        self.Q[state_idx][action] += self.learning_rate * td_error

    def decay_opt(self, opt_id):
        """
        Decay the option parameter for the specific option.
        The parameter is decayed using the decay factor and is bounded by the final parameter.
        """
        self.param_opt_lis[opt_id] = max(self.param_opt_lis[opt_id] * self.param_decay, self.final_param)

    def decay(self):
        """
        Decay the parameter for the agent.
        The parameter is decayed using the decay factor and is bounded by the final parameter.
        """
        self.param = max(self.param * self.param_decay, self.final_param)


class Option_Agent_alt:
    """
    Option Agent for the Taxi environment.
    This agent implements the options framework for reinforcement learning.
    It can learn both intra-option and inter-option policies.
    """
    def __init__(
            self, env, q_table, state_shape, action_shape, option_shape, action_map,
            policy, param, final_param, param_decay, discount_factor, learning_rate, algo
        ):
        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.option_shape = option_shape
        self.policy = policy
        self.param = param
        self.final_param = final_param
        self.param_decay = param_decay
        self.action_map = action_map
        self.Q = q_table
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.algo = algo
        self.Q_opt_lis = [np.zeros((5, 5, action_shape)) for _ in range(option_shape)]
        self.param_opt_lis = [param for _ in range(option_shape)]
        
    def decode(self, state):
        """
        Decode the state from the environment into its components.
        The state is represented as a single integer, and we decode it into
        taxi_row, taxi_column, passenger_location, and destination.
        """
        taxi_row = state // 100
        taxi_column = (state // 20) % 5
        passenger_location = (state % 20) // 4
        destination = (state % 20) % 4
        return (taxi_row, taxi_column, passenger_location, destination)

    def get_action(self, state_idx):
        """
        Get the action to take based on the current policy.
        The action can be either a normal action or an option.
        """
        actions = self.action_shape + self.option_shape
        state_decoded = self.decode(state_idx)
        
        # If the policy is epsilon-greedy, choose a random action with probability epsilon
        # Otherwise, choose the action with the highest Q-value
        if self.policy == 'epsilon_greedy':
            if np.random.random() < self.param:
                return np.random.randint(0, actions)
            else:
                return np.argmax(self.Q[state_idx][:actions])
        
        # If the policy is softmax, calculate the probabilities of each action
        # based on their Q-values and sample an action from the distribution
        
        elif self.policy == 'softmax':
            q_values = self.Q[state_idx][:actions] / self.param
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(actions, p=probabilities)
        
        return 0
    
    def get_opt_action(self, state_idx, opt_id):
        """
        Get the action to take based on the current policy for a specific option.
        The action can be either a normal action or an option.
        """
        actions = self.action_shape

        # If the policy is epsilon-greedy, choose a random action with probability epsilon
        # Otherwise, choose the action with the highest Q-value
        if self.policy == 'epsilon_greedy':
            if np.random.random() < self.param_opt_lis[opt_id]:
                return np.random.randint(0, actions)
            else:
                return np.argmax(self.Q_opt_lis[opt_id][state_idx][:actions])
        
        # If the policy is softmax, calculate the probabilities of each action
        # based on their Q-values and sample an action from the distribution
        elif self.policy == 'softmax':
            q_values = self.Q_opt_lis[opt_id][state_idx][:actions] / self.param_opt_lis[opt_id]
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(actions, p=probabilities)
        return 0

    # Check if the option is done based on the state
    def is_pickup_terminated(self, state_decode):
        return (state_decode[0] == 0 and state_decode[1] == 0) or (state_decode[0] == 4 and state_decode[1] == 0) or (state_decode[0] == 0 and state_decode[1] == 4) or (state_decode[0] == 4 and state_decode[1] == 3)

    def is_dropoff_terminated(self, state_decode):
        return (state_decode[0] == 0 and state_decode[1] == 0) or (state_decode[0] == 4 and state_decode[1] == 0) or (state_decode[0] == 0 and state_decode[1] == 4) or (state_decode[0] == 4 and state_decode[1] == 3)
    
    
    def Option(self, state_idx, option_id):
        """
        Execute the option in the environment.
        The option can be either a pickup or dropoff option.
        The option is executed until it is done or the episode terminates.
        """
        optdone = False
        state = state_idx
        tot_reward = 0
        undecay_reward = 0
        steps = 0
        terminated, truncated = False, False
        opt_idx = option_id - 6

        # Loop until the option is done or the episode terminates
        while not optdone and not terminated and not truncated:
            state_decode = self.decode(state)
            taxi_row, taxi_col = state_decode[0], state_decode[1]

            if option_id == 6 and self.is_pickup_terminated(state_decode):
                optdone = True
                
            if option_id == 7 and self.is_dropoff_terminated(state_decode):
                optdone = True
            

            # if option_id == 6:
            if optdone:
                
                if (self.is_pickup_terminated(state_decode) or self.is_dropoff_terminated(state_decode)):
                    if option_id == 6:
                        next_state, reward, terminated, truncated, _ = self.env.step(action=4)
                        optaction = 4
                    elif option_id == 7:
                        next_state, reward, terminated, truncated, _ = self.env.step(action=5)
                        optaction = 5
                    
                    next_state_decode = self.decode(state=next_state)
                    next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]

                    # Update the Q-values based on the algorithm
                    # If the algorithm is intra-option, update the Q-values for both the option and the action
                    if self.algo == 'intra_option':
                        self.update(state_idx=state, action=optaction, reward=reward,
                                    terminated=terminated, next_state_idx=next_state)
                        
                        # Update the Q-values for all options that took the same action
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                        self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                        for opt_id in range(self.option_shape):
                            
                            if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                                self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                                reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                                self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                    # If the algorithm is smdp, update the Q-values for the option only
                    elif self.algo == 'smdp':
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                    
                    undecay_reward += reward
                    tot_reward += reward * (self.discount_factor ** steps)
                    state = next_state
                    steps += 1

                else:
                    optaction = self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_idx)
                    next_state, reward, terminated, truncated, _ = self.env.step(action=optaction)
                    next_state_decode = self.decode(state=next_state)
                    next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]
                    # Update the Q-values based on the algorithm
                    # If the algorithm is intra-option, update the Q-values for both the option and the action
                    if self.algo == 'intra_option':
                        self.update(state_idx=state, action=optaction, reward=reward,
                                    terminated=terminated, next_state_idx=next_state)
                        
                        # Update the Q-values for all options that took the same action
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                        self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                        for opt_id in range(self.option_shape):
                            
                            if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                                self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                                reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                                self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                    # If the algorithm is smdp, update the Q-values for the option only
                    elif self.algo == 'smdp':
                        self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                        reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))

                    undecay_reward += reward
                    tot_reward += reward * (self.discount_factor ** steps)
                    steps += 1
                    state = next_state

            else:
            # Get the action for the option
            # Execute the action in the environment and get the next state and reward
                optaction = self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_idx)
                next_state, reward, terminated, truncated, _ = self.env.step(action=optaction)
                next_state_decode = self.decode(state=next_state)
                next_taxi_row, next_taxi_col = next_state_decode[0], next_state_decode[1]

                # Update the Q-values based on the algorithm
                # If the algorithm is intra-option, update the Q-values for both the option and the action
                if self.algo == 'intra_option':
                    self.update(state_idx=state, action=optaction, reward=reward,
                                terminated=terminated, next_state_idx=next_state)
                    
                    # Update the Q-values for all options that took the same action
                    self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                    reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                    self.update(state_idx = state, action = opt_idx + 6, reward = reward, terminated = terminated, next_state_idx = next_state)
                    for opt_id in range(self.option_shape):
                        
                        if self.get_opt_action(state_idx=(taxi_row, taxi_col), opt_id=opt_id) == optaction and opt_id != opt_idx:
                            self.update_opt(opt_id=opt_id, state_idx=(taxi_row, taxi_col), action=optaction,
                                            reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))
                            self.update(state_idx = state, action = opt_id + 6, reward = reward, terminated = terminated, next_state_idx = next_state)

                # If the algorithm is smdp, update the Q-values for the option only
                elif self.algo == 'smdp':
                    self.update_opt(opt_id=opt_idx, state_idx=(taxi_row, taxi_col), action=optaction,
                                    reward=reward, terminated=terminated, next_state_idx=(next_taxi_row, next_taxi_col))

                undecay_reward += reward
                tot_reward += reward * (self.discount_factor ** steps)
                steps += 1
                state = next_state

        # Decay the option parameter after the option is done
        self.decay_opt(opt_id=opt_idx)
        return tot_reward, undecay_reward, steps, state, terminated, truncated

    def update_opt(self, opt_id, state_idx, action, reward, terminated, next_state_idx):
        """
        Update the Q-values for the option based on the reward and next state.
        The Q-values are updated using the TD error.
        """
        q_next = 0 if terminated else np.max(self.Q_opt_lis[opt_id][next_state_idx])
        td_error = reward + self.discount_factor * q_next - self.Q_opt_lis[opt_id][state_idx][action]
        self.Q_opt_lis[opt_id][state_idx][action] += self.learning_rate * td_error

    def update(self, state_idx, action, reward, terminated, next_state_idx):
        """
        Update the Q-values for the action based on the reward and next state.
        The Q-values are updated using the TD error.
        """
        q_next = 0 if terminated else np.max(self.Q[next_state_idx])
        td_error = reward + self.discount_factor * q_next - self.Q[state_idx][action]
        self.Q[state_idx][action] += self.learning_rate * td_error

    def decay_opt(self, opt_id):
        """
        Decay the option parameter for the specific option.
        The parameter is decayed using the decay factor and is bounded by the final parameter.
        """
        self.param_opt_lis[opt_id] = max(self.param_opt_lis[opt_id] * self.param_decay, self.final_param)

    def decay(self):
        """
        Decay the parameter for the agent.
        The parameter is decayed using the decay factor and is bounded by the final parameter.
        """
        self.param = max(self.param * self.param_decay, self.final_param)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def plot_value_policy(agent, save_path=None, option_choice=1):
    action_names = agent.action_map
    nA = len(action_names)
    # Enhanced colormap with better visual distinction
    policy_cmap = ListedColormap(sns.color_palette("viridis", nA))
    
    loc_positions = {0:(0,0), 1:(0,4), 2:(4,0), 3:(4,3)}
    loc_names     = {0:'R', 1:'G', 2:'Y', 3:'B', 4:'In Taxi'}
    
    # Set consistent style - removing grid for policy plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Two figures: destinations [0,1] and [2,3]
    for fig_idx, dest_pair in enumerate([[0,1], [2,3]]):
        # Build valid pickup lists for each destination
        pickups_for_dest = []
        for d in dest_pair:
            pickups = [p for p in range(5) if not (p<4 and p==d)]
            pickups_for_dest.append(pickups)  # length 4 each
        
        # MODIFIED: Increased figure size and improved aspect ratio
        fig, axes = plt.subplots(4, 4, figsize=(22, 18))
        
        # MODIFIED: Improved spacing with more room at top and between subplots
        plt.subplots_adjust(top=0.9, wspace=0.05, hspace=0.2)
        fig.set_facecolor('white')
        
        for j, dest in enumerate(dest_pair):
            for i, pickup in enumerate(pickups_for_dest[j]):
                # Compute Q‐value grids
                V = np.zeros((5,5))
                P = np.zeros((5,5), dtype=int)
                for r in range(5):
                    for c in range(5):
                        s = r*100 + c*20 + pickup*4 + dest
                        q = agent.Q[s]
                        V[r,c] = q.max()
                        P[r,c] = q.argmax()
                
                ax_val = axes[i, j*2]
                ax_pol = axes[i, j*2 + 1]
                
                # Value heatmap with improved annotations
                sns.heatmap(
                    V, annot=True, fmt=".2f", cmap="viridis", 
                    annot_kws={"size": 11, "weight": "bold"},  # MODIFIED: Increased font size
                    cbar_kws={'label':'Max Q-value', 'shrink': 0.8}, ax=ax_val
                )
                ax_val.set_title(f"Pick {loc_names[pickup]} → Dest {loc_names[dest]} (Value)", 
                                 fontsize=14, fontweight='bold')  # MODIFIED: Increased font size
                ax_val.set_xlabel("Taxi Col", fontsize=12)
                ax_val.set_ylabel("Taxi Row", fontsize=12)
                
                # Policy image with better visualization - removing grid lines
                im = ax_pol.imshow(P, cmap=policy_cmap, vmin=0, vmax=nA-1)
                ax_pol.set_title(f"Pick {loc_names[pickup]} → Dest {loc_names[dest]} (Policy)", 
                                 fontsize=14, fontweight='bold')  # MODIFIED: Increased font size
                ax_pol.set_xlabel("Taxi Col", fontsize=12)
                ax_pol.set_ylabel("Taxi Row", fontsize=12)
                
                # Grid lines for policy plot - but ONLY for cell boundaries
                ax_pol.set_xticks(np.arange(-.5,5,1), minor=True)
                ax_pol.set_yticks(np.arange(-.5,5,1), minor=True)
                ax_pol.grid(which='minor', color='black', linestyle='-', linewidth=1)
                ax_pol.grid(which='major', visible=False)  # Remove default grid
                ax_pol.tick_params(which='minor', size=0)
                
                # Enhanced action text with better contrast
                for r in range(5):
                    for c in range(5):
                        a = P[r,c]
                        txt = action_names[a][:6]
                        # Dynamic text color for better readability
                        color = 'white' if policy_cmap(a/max(1, nA-1))[0] + policy_cmap(a/max(1, nA-1))[1] + policy_cmap(a/max(1, nA-1))[2] < 1.5 else 'black'
                        ax_pol.text(c, r, txt, ha='center', va='center',
                                    fontsize=10, fontweight='bold', color=color)  # MODIFIED: Increased font size
                
                # Highlight RGYB locations with enhanced styling
                for loc,(lr,lc) in loc_positions.items():
                    # Value plot locations - enhanced borders
                    ax_val.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                    edgecolor='black', linewidth=2))
                    if loc == dest:
                        ax_val.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                        edgecolor='red', linestyle='--', linewidth=2.5))
                    if loc == pickup and pickup<4:
                        ax_val.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                        edgecolor='green', linestyle='--', linewidth=2.5))
                    
                    # Enhanced location labels
                    bgcolor = ('red' if loc==dest else
                                'lightgreen' if loc==pickup and pickup<4 else
                                'lightyellow')
                    textcolor = 'white' if loc==dest else 'black'
                    
                    ax_val.text(lc+0.5, lr+0.2, loc_names[loc],
                            ha='center', va='center', fontsize=9, color=textcolor,  # MODIFIED: Increased font size
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=bgcolor, alpha=0.9, edgecolor='gray'),
                            zorder=10)
                
                # Policy plot locations
                for loc,(lr,lc) in loc_positions.items():
                    lr -= 0.5
                    lc -= 0.5

                    ax_pol.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                    edgecolor='black', linewidth=2))
                    if loc == dest:
                        ax_pol.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                        edgecolor='red', linestyle='--', linewidth=2.5))
                    if loc == pickup and pickup<4:
                        ax_pol.add_patch(mpatches.Rectangle((lc,lr),1,1, fill=False,
                                                        edgecolor='green', linestyle='--', linewidth=2.5))
                    
                    # Enhanced location labels - same as value plot
                    bgcolor = ('red' if loc==dest else
                                'lightgreen' if loc==pickup and pickup<4 else
                                'lightyellow')
                    textcolor = 'white' if loc==dest else 'black'
                    
                    ax_pol.text(lc+0.5, lr+0.2, loc_names[loc],
                            ha='center', va='center', fontsize=9, color=textcolor,  # MODIFIED: Increased font size
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=bgcolor, alpha=0.9, edgecolor='gray'),
                            zorder=10)
        
        # Enhanced main title with proper position - MODIFIED: Adjusted position
        fig.suptitle(f"Algorithm: {agent.algo}, Options: {agent.option_shape}", 
                     fontsize=24, fontweight='bold', y=0.95)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}/Algorithm_{agent.algo}_option_choice_{option_choice}_dest_{dest_pair[0]}_{dest_pair[1]}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def plot_options_value_policy(agent, save_path=None, option_choice=1):
    """
    Plot the value functions and policies for all options in a 5x5 grid.
    Each row represents one option.
    
    Parameters:
    -----------
    agent : Option_Agent or Option_Agent_alt
        The agent with trained option policies
    save_path : str, optional
        Path to save the plot, if None the plot will be displayed
    option_choice : int
        Option choice (1 or 2) to determine option names
    """
    
    # Get the number of options and action names
    option_shape = agent.option_shape
    action_names = {i: agent.action_map[i] for i in range(agent.action_shape)}  # Only primitive actions
    nA = agent.action_shape
    
    # Create a colormap for the policy
    policy_cmap = ListedColormap(sns.color_palette("viridis", nA))
    
    # Location positions and names
    loc_positions = {0:(0,0), 1:(0,4), 2:(4,0), 3:(4,3)}
    loc_names = {0:'R', 1:'G', 2:'Y', 3:'B', 4:'In Taxi'}
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with one row per option, two columns (value and policy)
    fig, axes = plt.subplots(option_shape, 2, figsize=(16, 5*option_shape))
    
    # If there's only one option, axes won't be a 2D array
    if option_shape == 1:
        axes = np.array([axes]).reshape(1, 2)
    
    # Improved spacing adjustments
    plt.subplots_adjust(hspace=0.25, wspace=0.05, top=0.9)
    fig.set_facecolor('white')
    
    # Option name based on the agent type
    if option_choice == 1:
        opt_names = ['reachR', 'reachG', 'reachY', 'reachB']
    else:
        opt_names = ['gopickup', 'godropoff']
    
    # Plot each option's value function and policy
    for opt_id in range(option_shape):
        # Extract the Q-values for this option
        Q_opt = agent.Q_opt_lis[opt_id]
        
        # Compute value and policy grids
        V = np.zeros((5, 5))
        P = np.zeros((5, 5), dtype=int)
        for r in range(5):
            for c in range(5):
                V[r, c] = np.max(Q_opt[(r, c)])
                P[r, c] = np.argmax(Q_opt[(r, c)])
        
        # Get axes for this option
        ax_val = axes[opt_id, 0]
        ax_pol = axes[opt_id, 1]
        
        # Value heatmap
        sns.heatmap(
            V, annot=True, fmt=".2f", cmap="viridis", 
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={'label': 'Max Q-value', 'shrink': 0.8}, ax=ax_val
        )
        ax_val.set_title(f"Option: {opt_names[opt_id]} (Value)", 
                        fontsize=14, fontweight='bold')
        ax_val.set_xlabel("Taxi Col", fontsize=12)
        ax_val.set_ylabel("Taxi Row", fontsize=12)
        
        # Policy image - positioned closer to the value heatmap
        im = ax_pol.imshow(P, cmap=policy_cmap, vmin=0, vmax=nA-1)
        ax_pol.set_title(f"Option: {opt_names[opt_id]} (Policy)", 
                        fontsize=14, fontweight='bold')
        ax_pol.set_xlabel("Taxi Col", fontsize=12)
        ax_pol.set_ylabel("Taxi Row", fontsize=12)
        
        # Grid lines for policy plot
        ax_pol.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax_pol.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax_pol.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax_pol.grid(which='major', visible=False)
        ax_pol.tick_params(which='minor', size=0)
        
        # Action text with good contrast
        for r in range(5):
            for c in range(5):
                a = P[r, c]
                txt = action_names[a][:6]  # Truncate long action names
                # Dynamic text color for readability
                color = 'white' if policy_cmap(a/max(1, nA-1))[0] + policy_cmap(a/max(1, nA-1))[1] + policy_cmap(a/max(1, nA-1))[2] < 1.5 else 'black'
                ax_pol.text(c, r, txt, ha='center', va='center',
                            fontsize=10, fontweight='bold', color=color)
        
        # Highlight RGYB locations
        for loc, (lr, lc) in loc_positions.items():
            # Value plot locations
            ax_val.add_patch(mpatches.Rectangle((lc, lr), 1, 1, fill=False,
                                            edgecolor='black', linewidth=2))
            
            # Policy plot locations
            lr_pol, lc_pol = lr - 0.5, lc - 0.5
            ax_pol.add_patch(mpatches.Rectangle((lc_pol, lr_pol), 1, 1, fill=False,
                                            edgecolor='black', linewidth=2))
            
            # Add location labels
            bgcolor = 'lightyellow'
            textcolor = 'black'
            
            ax_val.text(lc+0.5, lr+0.2, loc_names[loc],
                    ha='center', va='center', fontsize=9, color=textcolor,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bgcolor, alpha=0.9, edgecolor='gray'),
                    zorder=10)
            
            ax_pol.text(lc_pol+0.5, lr_pol+0.2, loc_names[loc],
                    ha='center', va='center', fontsize=9, color=textcolor,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bgcolor, alpha=0.9, edgecolor='gray'),
                    zorder=10)
    
    # Add a legend for actions
    handles = []
    for a in range(nA):
        color = policy_cmap(a/(nA-1))
        handles.append(mpatches.Patch(color=color, label=action_names[a]))
    
    # Main title - moved closer to the plots
    fig.suptitle(f"Algorithm: {agent.algo}, Option-choice: {option_choice}, Options' Value Functions and Policies", 
                fontsize=20, fontweight='bold', y=0.96)
    
    # Save or show the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}/Options_{agent.algo}_option_choice_{option_choice}.png", 
                  dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # Adjusted to account for title and legend
        plt.show()

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