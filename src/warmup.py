import gymnasium
import torch


# test policy, local to this class
class Policy:
    def __init__(self):
        self.model = torch.nn.Sequential(
                torch.nn.Linear(4,10), torch.nn.ReLU(), torch.nn.Linear(10, 2), torch.nn.Sigmoid(), torch.nn.Softmax())
    def get_action(self, observation):
        observation_tensor = torch.tensor(observation)
        action_probs = self.model(observation_tensor)
        sampled_action = torch.multinomial(action_probs, 1).item()
        return sampled_action
        

#create an environment
env = gymnasium.make('CartPole-v1', render_mode="human")


#create a policy
policy = Policy()


# get an observation from env. pass the obs to policy to get an action 
observation, info = env.reset()
action = policy.get_action(observation=observation)

#pass the action to the env to get reward and next observation
print(action)
observation, reward, terminated, truncated, info = env.step(action=action)
print(reward)