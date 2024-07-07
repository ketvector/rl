import gymnasium
import torch
import argparse

from policy import Policy

def run_trials(untrained = False):
    #create environment and policy
    env = gymnasium.make('CartPole-v1', render_mode= "human")
    policy = Policy()
    if not untrained:
        policy.load_state_dict(torch.load('./saved-models/vpg.pth'))

    obs, _ = env.reset()
    truncated, terminated = False, False
    num_iterations = 10
    reward_sum = 0
    for _ in range(num_iterations):
        env.reset()
        truncated, terminated = False, False
        total_reward = 0
        while truncated == False and terminated == False:
            action = torch.argmax(policy(obs)).item()
            obs, reward, truncated, terminated, _ = env.step(action)
            total_reward = total_reward + reward
        print(f"episode reward: {total_reward}")
        reward_sum = reward_sum + total_reward

    print(f"average reward : {reward_sum / num_iterations}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--untrained", action = "store_true")
    parser.add_argument("--num_runs", default=10,  type=int)
    args = parser.parse_args()
    untrained = args.untrained
    run_trials(untrained)
    




