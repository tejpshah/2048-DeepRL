
# standard imports 
import gym 
import numpy as np 

# pytorch deep learning framework 
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.distributions.categorical import Categorical 

# logging imports 
import json
import matplotlib.pyplot as plt

# set up device on which to update 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# sets up Actor Critic network for policy/value functions 
class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_size=16,
        act_size=4,
        hidden_layer_size=128,
        num_shared_layers=2,
        activation_function=nn.Tanh(),
    ):
        super().__init__()
        
        # Initialize instance variables
        self.hidden_layer_size = hidden_layer_size
        self.num_shared_layers = num_shared_layers
        self.activation_function = activation_function

        # Create shared layers
        shared_layers = []
        for i in range(num_shared_layers):
            in_size = obs_size if i == 0 else hidden_layer_size
            out_size = hidden_layer_size
            shared_layers.append(nn.Linear(in_size, out_size))
            shared_layers.append(self.activation_function)
        self.shared_layers = nn.Sequential(*shared_layers)

        # Create policy and value layers
        self.policy_layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            self.activation_function,
            nn.Linear(hidden_layer_size, act_size),
        )

        self.value_layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            self.activation_function,
            nn.Linear(hidden_layer_size, 1),
        )

    def policy_function(self, state):
        # Compute the policy output
        return self.policy_layers(self.shared_layers(state))

    def value_function(self, state):
        # Compute the value output
        return self.value_layers(self.shared_layers(state))

    def forward(self, state):
        # Compute both policy and value outputs
        return self.policy_function(state), self.value_function(state)

# sets up PPO trainer that updates the weights of Actor Critic
class PPO_Trainer():

  def __init__(self, actor_critic, ppo_clip_val = 0.2, ppo_lr = 3e-4, val_lr = 1e-3, ppo_epochs=16, val_epochs=16, kl_earlystopping=0.01):
    
    # Initialize instance variables
    self.actor_critic = actor_critic
    self.ppo_clip_val = ppo_clip_val 
    self.ppo_epochs = ppo_epochs
    self.val_epochs = val_epochs 
    self.kl_bound = kl_earlystopping 

    # Set up optimizers
    policy_params = list(self.actor_critic.shared_layers.parameters()) + list(self.actor_critic.policy_layers.parameters())
    value_params = list(self.actor_critic.shared_layers.parameters()) + list(self.actor_critic.value_layers.parameters())

    self.policy_optim = optim.Adam(policy_params, lr=ppo_lr)
    self.value_optim = optim.Adam(value_params, lr=val_lr)
  
  def train_policy(self, states, actions, old_log_probs, gaes):
    
    # Train the policy network using PPO
    for _ in range(self.ppo_epochs):
      self.policy_optim.zero_grad() 

      # Compute new log probabilities and ratios
      new_logits = self.actor_critic.policy_function(states)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(actions)
      ratio = torch.exp(new_log_probs - old_log_probs)

      # Compute PPO loss and take an optimization step
      clipped_ratio = ratio.clamp(1-self.ppo_clip_val, 1+self.ppo_clip_val)
      ppo_loss = -torch.min(ratio*gaes, clipped_ratio*gaes).mean()
      ppo_loss.backward()
      self.policy_optim.step()

      # Check early stopping condition
      approx_kl_div = (old_log_probs - new_log_probs).mean()
      if self.kl_bound < approx_kl_div:
        break

  def train_value(self, states, returns):
    
    # Train the value network using MSE loss
    for _ in range(self.val_epochs):
      self.value_optim.zero_grad() 

      # Compute value loss and take an optimization step
      values = self.actor_critic.value_function(states)
      value_loss = (returns - values).pow(2).mean()
      value_loss.backward()
      self.value_optim.step()

# sets up PPO buffer to collect data and enable agent to act in MDP
class PPO_Buffer():

  def compute_discounted_rewards(self, rewards, gamma=0.99):
    """
    Computes the discounted rewards for a given sequence of rewards using the specified discount factor.
    """

    # Initialize the list of discounted rewards with the reward obtained at the last time step.
    new_rewards = [float(rewards[-1])]

    # Iterate over the time steps in reverse order, starting from the second to last time step.
    for i in reversed(range(len(rewards)-1)):

        # Compute the discounted reward for the current time step by adding the reward obtained at the current time step
        # to the discounted reward obtained at the next time step, multiplied by the discount factor gamma.
        discounted_reward = float(rewards[i]) + gamma * new_rewards[-1]

        # Append the computed discounted reward to the list of discounted rewards.
        new_rewards.append(discounted_reward)

    # Reverse the order of the list of discounted rewards and convert it to a numpy array.
    return np.array(new_rewards[::-1])

  def compute_advantages_gae(self, rewards, values, gamma=0.99, decay=0.95):
    """
    Calculate the Generalized Advantage Estimation (GAE) for a given sequence of rewards and corresponding predicted values.
    """

    # Calculate the predicted value of the next state for each time step.
    next_values = np.concatenate([values[1:], [0]])

    # Compute the temporal difference (TD) errors for each time step.
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    # Compute the GAEs for each time step.
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    # Return the computed GAEs as a numpy array.
    return np.array(gaes[::-1])

  def randomize_training_data_order(self, buffer):
      '''
      Randomizes the order of the training data in the buffer.
      '''

      # Create a list of indices
      permute_indices = np.random.permutation(len(buffer[0]))

      # Update the buffer with the randomized order
      states = torch.tensor(buffer[0][permute_indices], dtype=torch.float32, device=DEVICE)
      actions = torch.tensor(buffer[1][permute_indices], dtype=torch.float32, device=DEVICE)
      returns = self.compute_discounted_rewards(buffer[2][permute_indices], gamma=0.99)
      returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
      gaes = torch.tensor(buffer[3][permute_indices], dtype=torch.float32, device=DEVICE)
      log_probs = torch.tensor(buffer[4][permute_indices], dtype=torch.float32, device=DEVICE)

      # Return the randomized buffer
      return states, actions, returns, gaes, log_probs
  
  def generate_n_rollouts(self, model, env, max_steps=1000, n=4):
      """
      Performs n rollouts using the specified model and environment, and computes GAE advantages.
      Returns training data in the shape (n_steps, observation_shape) and the cumulative reward.
      """
      train_data = [[], [], [], [], []]  # Initialize lists to store training data
      ep_reward = 0  # Initialize cumulative reward
      for _ in range(n):
        obs = env.reset()
        for _ in range(max_steps):
            # Take an action according to the policy and record the results
            logits, val = model(torch.tensor([obs], dtype=torch.float32,device=DEVICE))
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()
            act, val = act.item(), val.item()
            next_obs, reward, done, _ = env.step(act)

            # Store the training data for this time step
            for i, item in enumerate((obs, act, reward, val, act_log_prob)):
              train_data[i].append(item)

            obs = next_obs
            ep_reward += reward
            if done:
                break

      # Convert the training data to numpy arrays and compute the GAE advantages
      train_data = [np.asarray(x) for x in train_data]
      train_data[3] = self.compute_advantages_gae(train_data[2], train_data[3])

      # Return the training data and the cumulative reward
      return train_data, ep_reward / n

