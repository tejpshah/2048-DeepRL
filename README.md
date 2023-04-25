# 2048-DeepRL
CS462 Final Project: Tej, Gloria &amp; Max -- 2048 with Deep RL

## Introduction and Motivation
This repository contains code for training DDQN (Double Deep Q Network) and PPO (Proximal Policy Optimization) agents to play the game 2048, as well as code for our 2048 game environment, plotting and visualization functions, and the agents' implementation in Cart Pole. It is our semester long project for the class CS 462, Introduction to Deep Learning, at Rutgers University New Brunswick. Our goal was to learn and try out DDQN and PPO implementations, which are promising algorithms in the field of deep reinforcement learning. We chose to have the agents learn to play the game 2048 because the game movements and objectives are simple to understand, but there are elements of randomness and a vast state space. 

## Game, Simulator, Other Work
We built our game emulating the [original 2048](https://play2048.co/). Movements are made by going left, right, up, and down, combining tiles to make larger tiles, where all tiles have values that are powers of two. When a tile of value 2048 is reached, the game is considered won. We tried to optimize the largest tile the agent could achieve on a game board, but also kept track of the game score. We have a simulator that uses the python multiprocessing library to run many simulations at once. There is a Plotter class in the utils folder, and gameplay visualization is done with the genVideo file. We also have a random agent, which at each state would choose randomly between the four possible actions, to serve as a baseline to compare our trained agents.

Prior to testing the agents on 2048, we trained models in the Cart Pole v1 environment; both models trained agents that obtained the optimal reward/score in Cart Pole. Notebooks for these files are also included here.

## Reward and State Representation
The DDQN model used the number of empty (0) tiles on the board to calculate its reward. The PPO model used the maximum tile on the board and the change in game score. 

For state representation, both models transformed the 4x4 game board into one hot vectors, corresponding to the value of the tile.  

## DDQN Agent and Results
DDQN is implemented by having two networks, a target network and an online network. The online network is updated using gradient descent, and chooses the best action to take. However, the target network is updated only periodically by copying the online network's weights, and it is used to help calculate the loss of an action (evaluate Q-value) taken by the online network. Having two networks reduces overestimation bias of Q-values, essentially reducing the problem of trying to optimize to a moving target. 

After training, the DDQN agent could win several games, achieving tiles valued at 2048. Most freqeuntly, the DDQN agent reached a maximum tile of 1024 before the board fills up. 

![](https://github.com/tejpshah/2048-DeepRL/blob/main/gifs/DDQN.gif)

Here is a bar graph of the maximum tile achieved by the DDQN agent in 1000 simulations. 

![](https://github.com/tejpshah/2048-DeepRL/blob/main/submission/ddqn/successful-model-2048/hd_ddqn2048_max_scores_bar.png)

## PPO Agent and Results
PPO is implemented as an improvement to TRPO. It optimizes a policy for the agent, but it essentially "clips" how far in one direction the policy will change, so that any changes to the policy are more conservative. We use the Actor-Critic Model; we have two networks, with some shared layers, one for the actor (policy) and one for the critic (value). The critic is used to evaluate actions proposed by the actor. 

After training, the PPO agent could achieve tiles of value 512 in a couple of games, but most frequently, it reached a maximum tile of 256 before the board fills up. 

![](https://github.com/tejpshah/2048-DeepRL/blob/main/gifs/PPO.gif)

Here is a bar graph of the maximum tile achieved by the PPO agent in 1000 simulations. 

![](https://github.com/tejpshah/2048-DeepRL/blob/main/submission/ppo/final-model-2048/final-max_scores_bar.png)

## Comparison to Random Agent
When testing our random agent, to serve as comparison for our trained agent, the random one most frequently got tiles of value 64. Here is a bar graph of the maximum tile achieved by the random agent in 1000 simulations. 

![](https://github.com/tejpshah/2048-DeepRL/blob/main/submission/random/random2048_max_scores_bar.png)

## Conclusion and Insights
Both DDQN and PPO performed better than the random agent, so both models do result in learning. The DDQN performed better than the PPO agent. Both models deserve further study. 
