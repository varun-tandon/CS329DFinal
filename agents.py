import random

class BaseAgent():
    def __init__(n_observations, n_actions):
        self.n_observations = n_observations
        self.n_actions = n_actions
    
    def act(self, state):
        raise NotImplementedError
    
    def update(self, state, action, next_state, reward):
        raise NotImplementedError

class DoubleDQNAgent(BaseAgent):
    def __init__(n_observations, n_actions):
        super.__init__(n_observations, n_actions)
        self.total_steps = 0
    
    def act(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.total_steps / EPS_DECAY)
        self.total_steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
