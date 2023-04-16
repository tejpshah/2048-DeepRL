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

# Hyperparameters for model
SHARED_HIDDEN_LAYER_SIZE= 64
NUM_SHARED_LAYERS = 1
ACTIVATION = nn.ReLU()
PPO_CLIP_VAL = 0.20
PPO_POLICY_LR = 3e-4
PPO_VALUE_LR = 3e-3
PPO_EPOCHS = 40
VAL_EPOCHS = 20
KL_TARGET = 0.02
N_EPISODES = 1000
PRINT_FREQ = 1
NUM_ROLLOUTS = 4
SAVE_FREQ = 50 

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

### HELPER FUNCTIONS ###

def train_ppo(env, model, ppo_trainer, ppo_buffer, n_episodes=N_EPISODES, num_rollouts=NUM_ROLLOUTS,
              print_freq=PRINT_FREQ, save_freq=SAVE_FREQ, save_model=True, model_path="cartpole_model",
              stats_path="cartpole_stats.json", load_from_checkpoint=False):
    """
    Trains a PPO model on a given environment using the provided trainer and buffer.

    Args:
    - env (gym.Env): the environment to train the model on
    - model (nn.Module): the PPO model to be trained
    - ppo_trainer (PPOTrainer): the PPO trainer used for training
    - ppo_buffer (PPOBuffer): the PPO buffer used for collecting training data
    - n_episodes (int): the number of episodes to run for (default: N_EPISODES)
    - num_rollouts (int): the number of rollouts to generate per episode (default: NUM_ROLLOUTS)
    - print_freq (int): how often to print episode statistics (default: PRINT_FREQ)
    - save_freq (int): how often to save the model (default: SAVE_FREQ)
    - save_model (bool): whether to save the model or not (default: True)
    - model_path (str): the path to save the model to (default: "cartpole_model")
    - stats_path (str): the path to save the episode statistics to (default: "cartpole_stats.json")
    - load_from_checkpoint (bool): whether to load from a checkpoint (default: False)

    Returns:
    - None
    """

    # Initialize variables
    num_steps = 0
    ep_rewards = []
    stats = {"avg_reward": [], "num_steps": []}

    # Load from checkpoint if flag is set
    if load_from_checkpoint:
        num_steps = 84284936
        with open('ppo_2048_stats_rewardfinal13.json') as f:
            stats = json.load(f)

    # Run the training loop
    for step in range(n_episodes):

        # Generate rollouts and collect training data
        train_data, reward = ppo_buffer.generate_n_rollouts(model, env, n=num_rollouts)
        ep_rewards.append(reward)

        # Randomize the order of the training data
        states, actions, rewards, gaes, log_probs = ppo_buffer.randomize_training_data_order(train_data)

        # Train the PPO model
        ppo_trainer.train_policy(states, actions, log_probs, gaes)
        ppo_trainer.train_value(states, rewards)

        # Update number of steps taken
        num_steps += len(train_data[0])

        # Print statistics every `print_freq` episodes
        if (step + 1) % print_freq == 0:
            avg_reward = np.mean(ep_rewards[-print_freq:])
            stats["avg_reward"].append(avg_reward)
            stats["num_steps"].append(num_steps)

            print(f"Episode {step+1} | Avg Reward {avg_reward:.1f} | NumSteps {num_steps}")

            # Save model every `save_freq` episodes
            if save_model and (step + 1) % save_freq == 0:
                torch.save(model.state_dict(), f"{model_path}_{step+1}.pt")

            # Save statistics to JSON file
            with open(stats_path, "w") as f:
                json.dump(stats, f)

def plot_training_stats(stats_file='cartpole_stats.json', w_size=20, dpi=300):
    """
    Generates a line plot of the average reward over the number of steps taken during training.

    Args:
    - stats_file (str): the path to the JSON file containing the training statistics (default: 'cartpole_stats.json')
    - w_size (int): the window size for smoothing the data (default: 20)
    - dpi (int): the resolution of the saved image (default: 300)

    Returns:
    - None
    """
    # Load the statistics from the JSON file
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Smooth the data with a rolling average
    window_size = w_size 
    rolling_avg_reward = np.convolve(stats["avg_reward"], np.ones(window_size)/window_size, mode='valid')
    rolling_num_steps = stats["num_steps"][window_size-1:]

    # Generate the plot
    plt.plot(rolling_num_steps, rolling_avg_reward)
    plt.xlabel("NumSteps")
    plt.ylabel("Avg Reward")
    plt.title("PPO CartPole Training")
    plt.savefig("ppo_cartpole_training_smooth.png", dpi=dpi)
    plt.show()

def evaluate_trained_model(model_path, env_name, num_episodes=1000):
    """
    Evaluates a trained PPO model on the specified environment using the saved model weights.

    Args:
    - model_path (str): the path to the saved model weights
    - env_name (str): the name of the environment to evaluate the model on
    - num_episodes (int): the number of episodes to run for (default: 1000)

    Returns:
    - avg_reward (float): the average reward received over the evaluation episodes
    """

    # Set up the environment
    env = gym.make(env_name)

    # Set up the model
    model = ActorCritic(env.observation_space.shape[0], 
                        env.action_space.n, 
                        hidden_layer_size=SHARED_HIDDEN_LAYER_SIZE, 
                        num_shared_layers=NUM_SHARED_LAYERS, 
                        activation_function=ACTIVATION)
    model = model.to(DEVICE)

    # Load saved model weights
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            logits, _ = model(torch.tensor([obs], dtype=torch.float32, device=DEVICE))
            act = torch.argmax(logits, dim=1).item()
            obs, reward, done, _ = env.step(act)
            total_reward += reward
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    print("Average reward:", avg_reward)

    return avg_reward

def plot_2048_training(stats_file='ppo_2048_stats.json', w_size=20, dpi=300):
    """
    Generate a line plot of the average reward over the number of steps taken during training.
    """
    with open(stats_file, "r") as f:
        stats = json.load(f)

    window_size = w_size 
    rolling_avg_reward = np.convolve(stats["avg_reward"], np.ones(window_size)/window_size, mode='valid')
    rolling_num_steps = stats["num_steps"][window_size-1:]

    plt.plot(rolling_num_steps, rolling_avg_reward)
    plt.xlabel("NumSteps")
    plt.ylabel("Avg Reward")
    plt.title("PPO 2048 Training")
    plt.savefig("ppo_2048_training_smooth.png", dpi=dpi)
    plt.show()

# this initializes the agent to use the game
class AgentPPO():
    """
    A PPO agent that selects actions based on a trained neural network.

    Attributes:
    - device (str): the device to run the model on (e.g. 'cpu' or 'cuda')
    - actions (np.ndarray): an array of the possible actions that can be taken
    - model (ActorCritic): the PPO model used to select actions
    - prev_state (np.ndarray): the previous state that was observed
    - count (int): the number of times the same state has been observed consecutively

    Methods:
    - choose_action(state) -> int: chooses an action to take based on the current state
    """
    def __init__(self, obs_space_size, act_space_size, hidden_layer_size, num_shared_layers, activation_function, device, model_path='ppo_2048_model.th'):
        """
        Initializes the AgentPPO instance with the specified parameters.

        Args:
        - obs_space_size (int): the size of the observation space
        - act_space_size (int): the size of the action space
        - hidden_layer_size (int): the size of the hidden layers in the model
        - num_shared_layers (int): the number of shared layers in the model
        - activation_function (function): the activation function used in the model
        - device (str): the device to run the model on (e.g. 'cpu' or 'cuda')
        - model_path (str): the path to the saved model weights (default: 'ppo_2048_model.th')
        """
        self.device = device
        self.actions = np.array([0, 1, 2, 3])

        # Create and load the PPO model
        self.model = ActorCritic(obs_space_size, act_space_size, hidden_layer_size, num_shared_layers, activation_function)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.prev_state = None
        self.count = 0

    def choose_action(self, state):
        """
        Chooses an action to take based on the current state.

        Args:
        - state (np.ndarray): the current state

        Returns:
        - act (int): the action to take
        """
        # If the same state has been observed more than 5 times, choose a random action
        if self.prev_state is not None and (self.prev_state == state).all():
            self.count += 1
        else:
            self.prev_state = state.copy()
            self.count = 0
        
        if self.count > 5:
            return np.random.choice(self.actions)
        else:
            # Preprocess the state and pass it through the model to get the logits
            state = np.log2(state, out=np.zeros_like(state), where=(state != 0)).reshape(-1).astype(int)
            state = np.eye(18)[state]
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

            logits, _ = self.model(torch.tensor(state.flatten(), dtype=torch.float32, device=self.device))
            act = torch.argmax(logits).item()
            return act

if __name__ == "__main__":
  
  ###  TRAINS MODEL USING PROXIMAL POLICY OPTIMIZATION FOR CARTPOLE ###

  # set up environment
  env = gym.make('CartPole-v1')

  # set up model
  model = ActorCritic(env.observation_space.shape[0], 
                      env.action_space.n, 
                      hidden_layer_size=SHARED_HIDDEN_LAYER_SIZE, 
                      num_shared_layers=NUM_SHARED_LAYERS, 
                      activation_function=ACTIVATION)
  model = model.to(DEVICE)  

  # set up PPO trainer
  ppo = PPO_Trainer(
      actor_critic = model, 
      ppo_clip_val = PPO_CLIP_VAL,
      ppo_lr = PPO_POLICY_LR,
      val_lr = PPO_VALUE_LR,
      ppo_epochs = PPO_EPOCHS, 
      val_epochs = VAL_EPOCHS,
      kl_earlystopping = KL_TARGET
  )

  # set up buffer
  ppobuffer = PPO_Buffer() 

  # train the model with PPO
  train_ppo(env=env, model=model, ppo_trainer=ppo, ppo_buffer = ppobuffer, save_freq=200)

  ###  PLOTS TRAINING AND EVALUATES TRAINED MODEL FOR PROXIMAL POLICY OPTIMIZATION ###

  # plot the training cartpole stats
  plot_training_stats('cartpole_stats.json')

  # evaluate the model
  evaluate_trained_model(model_path="cartpole_model_1000.pt", env_name = 'CartPole-v1', num_episodes=1000)

