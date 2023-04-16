import json
import numpy as np 
import matplotlib.pyplot as plt

def plot_2048_training(stats_file='models/ppo_2048_stats_rewardfinal.json', w_size=5000, dpi=300):
    """
    Generate a line plot of the average reward over the number of steps taken during training.
    
    Args:
    stats_file (str): The file path to the JSON file containing the training statistics.
    w_size (int): The window size used for the rolling average calculation.
    dpi (int): The resolution of the saved figure in dots per inch.

    Returns:
    None
    """
    # Load the JSON file with the training statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Calculate the rolling average
    window_size = w_size 
    rolling_avg_reward = np.convolve(stats["avg_reward"], np.ones(window_size)/window_size, mode='valid')
    rolling_num_steps = stats["num_steps"][window_size-1:]

    # Plot the rolling average over the number of steps
    plt.plot(rolling_num_steps, rolling_avg_reward)
    plt.xlabel("NumSteps")
    plt.ylabel("Avg Reward")
    plt.title("PPO 2048 Training")

    # Save the figure and show it
    plt.savefig("ppo_2048_training.png", dpi=dpi)
    plt.show()

# Example usage of the function
plot_2048_training('models/ppo_2048_stats_rewardfinal15.json')
