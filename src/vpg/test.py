import gymnasium
import torch
from policy import Policy


#create environment and policy
env = gymnasium.make('CartPole-v1', render_mode= "human")
policy = Policy()
policy.load_state_dict(torch.load('./saved-models/vpg.pth'))

obs, info = env.reset()
truncated, terminated = False, False
num_iterations = 10
reward_sum = 0
for i in range(num_iterations):
    observation, info = env.reset()
    truncated, terminated = False, False
    total_reward = 0
    while truncated == False and terminated == False:
        action = torch.argmax(policy(obs)).item()
        obs, reward, truncated, terminated, _ = env.step(action)
        total_reward = total_reward + 1
    print(f"episode reward: {total_reward}")
    reward_sum = reward_sum + total_reward

print(f"average reward : {reward_sum / num_iterations}")
    




