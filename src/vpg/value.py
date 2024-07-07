import torch

# approximator for value function to use as baseline reward    
class BaselineValue(torch.nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 20):
        super(BaselineValue, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim,1), torch.nn.ReLU())
    
    def forward(self, observation):
        observation_tensor = torch.tensor(observation)
        expected_reward = self.model(observation_tensor)
        return expected_reward