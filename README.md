# 2048-DeepRL
CS462 Final Project: Tej, Gloria &amp; Max -- 2048 with Deep RL

# Notes (Feb 28th by Tej)
- Added speed boost to simulator class with multiprocessing.
- Added the random agent uniformly selecting actions
- We should make all the agents inherit from this random class for good OOP (?)
- Added functionality to run n simulations for an agent playing 2048 with statistics in dict (max_score, num_steps, game_score)
- (Considerations for later) -- might need to update select_action in simulator depending on our DQN/DDQN/PPO/TRPO agent implementations
- (Considerations for later) -- might need to look into multithreading and processing so we can train RL Agents and run simulations much faster 
- Document any questions or comments over here as you work on it so we have a working log of the key things to focus on  

# LIST OF TASKS WEEK 1

## Environment Simulator 
NOTE: All subject to what you think is the best or if you have any better ideas - these are just some of the ideas I came up with. 
- TODO: @Gloria & Max - Sanity check 2048 numpy implementation. Add any additional useful functions you think might be necessary.
- TODO: @Gloria & Max - Sanity check 2048 CLI implementation. Add any additional useful functions you think might be necessary.
- TODO: @Gloria & Max - Any thoughts on Software Design? How should we structure the files / DeepRL techniques / enviornment / logging techniques?
- TODO: @Gloria & Max - Any other suggestions / thoughts on the simulator? Any lingering questions? Any better implementation strategies?

## Random Agent Baseline & General Agent Functions
NOTE: All subject to what you think is the best or if you have any better ideas - these are just some of the ideas I came up with. 
- TODO: @Gloria/Max/Tej - Build random agent (unfiormly samples action) that interacts with the Game class which interacts with the 2048 Board Class. 
- TODO: @Gloria/Max/Tej - Build a Simulation Class or Script that runs a selected agent n times and then keeps track of the game score, the max score, and number of steps in each game.
- TODO: @Gloria/Max/Tej - Build utility function to plot histogram of max game score over n simulations for a particular agent (+ saves the image as JPG). 
- TODO: @Gloria/Max/Tej - Build utility function to plot histogram of overall game score over n simulations for a particular agent (+ saves the image as JPG). 
- TODO: @Gloria/Max/Tej - Build utility function to automatically save game_score, max_score, and num_steps for a particular simulation to a JSON file. 
- TODO: @Gloria/Max/Tej - Build a visualization of the 2048 game so that we can see and play the game well and debug faster later on.
- TODO: @Gloria/Max/Tej - Build a visualization that can generate a short video of the 2048 game sequence given an episode trajectory. Useful for debugging
