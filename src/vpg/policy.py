import torch
from torch import nn
# function approximator for policy
class Policy(torch.nn.Module):
    def __init__(self, dims = [4,10,2]):
        super(Policy, self).__init__()
        module_list = nn.ModuleList([])
        in_out = list(zip(dims[:-1], dims[1:]))
        for idx, item in enumerate(in_out):
            activation = torch.nn.ReLU() if (idx != len(in_out) - 1) else torch.nn.Sigmoid()
            module_list.extend(nn.ModuleList([torch.nn.Linear(item[0], item[1]), activation]))
        module_list.extend(nn.ModuleList([torch.nn.Softmax()]))
        self.model = torch.nn.Sequential(*module_list)
    def forward(self, observation):
        observation_tensor = torch.tensor(observation)
        action_probs = self.model(observation_tensor)
        return action_probs