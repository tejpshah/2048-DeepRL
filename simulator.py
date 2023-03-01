import time 
from models.env.board import Board
from models.agent_random import AgentRandom

class Simulator():
    
    def __init__(self, agent=AgentRandom()):

        # selects which agent to run simulations from
        self.agent = agent 

        # stores simulation info for plotting
        self.game_scores = dict()
        self.max_scores = dict() 
        self.num_steps = dict() 
    
    def run_episode(self):

        game = Board()

        n_steps = 0 

        # runs an episode until termination 
        while not game.is_terminal_state():

            # gets state, and makes action based on state
            state = game.get_state() 
            action = self.agent.choose_action(state)

            # updates game board steps
            game.move(action)
            n_steps += 1

        # updates simulation info dictionaries 
        self.num_steps[n_steps] = self.num_steps.get(n_steps, 0) + 1
        self.max_scores[game.get_max()] = self.max_scores.get(game.get_max(), 0) + 1
        self.game_scores[game.score] = self.game_scores.get(game.score, 0) + 1
    
    def run_episodes(self, num_episodes=1000):
        start = time.time() 
        for _ in range(num_episodes):
            self.run_episode()
        end = time.time() 
        print(f"It took {end-start:.3f} seconds to run {num_episodes} simulations.")

    def get_simulation_info(self):
        print('\nTHIS WAS THE SIMULATION INFO:')
        print(f"Game Scores Dictionary: {self.game_scores}")
        print(f"Max Scores Dictionary: {self.max_scores}")
        print(f"Episode Steps Dictionary: {self.num_steps}\n")

if __name__ == "__main__":
    S1 = Simulator()
    S1.run_episodes()
    # S1.get_simulation_info()