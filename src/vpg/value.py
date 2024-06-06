import torch

# approximator for value function to use as baseline reward    
class BaselineValue(torch.nn.Module):
    # TODO: take model input/output at model creation
    def __init__(self):
        super(BaselineValue, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 20), torch.nn.ReLU(), torch.nn.Linear(20,1), torch.nn.ReLU())
    
    def forward(self, observation):
        observation_tensor = torch.tensor(observation)
        expected_reward = self.model(observation_tensor)
        return expected_reward