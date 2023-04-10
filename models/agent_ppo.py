import torch 
from . import train_ppo_base

class AgentPPO():
    def __init__(self, obs_space_size, act_space_size, hidden_layer_size, num_shared_layers, activation_function, device, model_path = 'ppo_2048_model.th'):
        self.device = device
        self.actions = np.array([0,1,2,3])

        self.model = ActorCritic(obs_space_size,act_space_size, hidden_layer_size, num_shared_layers, activation_function)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location = self.device))

        self.prev_state = None
        self.count = 0

    def choose_action(self, state):

        if self.prev_state is not None and (self.prev_state == state).all():
           self.count += 1
        else:
           self.prev_state = state.copy()
           self.count = 0
        
        if self.count > 5:
            return np.random.choice(self.actions)
        else: 
            state = np.log2(state, out=np.zeros_like(state), where=(state != 0)).reshape(-1).astype(int)
            state = np.eye(18)[state]
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

            logits, _ = self.model(torch.tensor([state.flatten()], dtype=torch.float32, device=self.device))
            act = torch.argmax(logits, dim=1).item()
            return act 