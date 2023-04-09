from wrapper import EnvironmentWrapper
from train_ppo_base import ActorCritic, PPO_Buffer, PPO_Trainer

if __name__ == "__main__":
    env = EnvironmentWrapper() 

    print(env.board.state)

    env.step('W')
    print(env.board.state)

    env.step('D')
    print(env.board.state)

    env.step('S')
    print(env.board.state)

    env.reset()
    print(env.board.state)




