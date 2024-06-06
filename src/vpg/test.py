import gymnasium
import torch
from policy import Policy


#create environment and policy
env = gymnasium.make('CartPole-v1', render_mode= "human")
policy = Policy()
policy.load_state_dict(torch.load('./saved-models/vpg.pth'))

obs, info = env.reset()
truncated, terminated = False, False

while truncated == False and terminated == False:
    action = torch.argmax(policy(obs)).item()
    obs, _, truncated, terminated, _ = env.step(action) 



