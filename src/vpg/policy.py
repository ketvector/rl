import torch

# function approximator for policy
class Policy(torch.nn.Module):
    #TODO: take model input/output at model creation
    def __init__(self):
        super(Policy, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4,10), torch.nn.ReLU(), torch.nn.Linear(10,2), torch.nn.Sigmoid(), torch.nn.Softmax())
    
    def forward(self, observation):
        observation_tensor = torch.tensor(observation)
        action_probs = self.model(observation_tensor)
        return action_probs