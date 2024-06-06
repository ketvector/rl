import torch
import gymnasium
from policy import Policy
from value import BaselineValue
from torch.utils.tensorboard import SummaryWriter


"""
Implementation of a policy gradient RL algorithm.  

"""
        
# calculates the policy loss    
def policy_loss(log_probs, weights):
    return  -1 * (log_probs * weights).mean()

# loss for value approximator
def value_loss(actual_reward, expected_reward):
    l = torch.nn.MSELoss()
    return l(actual_reward, expected_reward)
    

def reward_to_go(episode_rewards):
    n = len(episode_rewards)
    rtg = torch.ones(n)
    sum = 0
    for i in reversed(range(n)):
        sum = sum + episode_rewards[i]
        rtg[i] = sum
    return rtg

# hyperparameters for training
epochs = 200
episodes_per_epoch = 100
policy_lr = 1e-2
value_lr = 1e-2


#create an environment
env = gymnasium.make('CartPole-v1')

#create networks for policy and value approximator
policy = Policy()
value = BaselineValue()
policy.zero_grad()
value.zero_grad()
policy_optimizer = torch.optim.Adam(policy.parameters(),lr=policy_lr)
value_optimizer = torch.optim.Adam(value.parameters(), lr=value_lr)

#get a summary writer for logging to tensorboard
writer = SummaryWriter()

for epoch in range(epochs):
    batch_log_probs = [] #collected log probs for each batch
    batch_rewards_to_go = [] #collected rewards to go for each batch 
    batch_expected_rewards = [] #predicted by our value approximator
    avg_reward = 0 #for performance tracking
    for episode in range(episodes_per_epoch):
        episode_log_probs = []
        episode_rewards = []
        episode_expected_rewards = []
        observation, info = env.reset()
        truncated, terminated = False, False
        action_prob_diff_avg = 0
        step = 0
        while truncated == False and terminated == False:
            action_probs = policy(observation) # get probablity for actions from our policy
            action = torch.multinomial(action_probs, 1).item() # sample from the probability distribution
            action_log_prob = torch.log(action_probs[action]) # select the sampled action
            observation, reward, terminated, truncated, info = env.step(action=action) # take the action. get rewards, next-observation
            episode_rewards.append(reward)
            episode_log_probs.append(action_log_prob)
            episode_expected_rewards.append(value(observation))
            action_prob_diff = action_probs[1] - action_probs[0]
            action_prob_diff_avg = ((action_prob_diff_avg * step) + action_prob_diff ) / (step + 1)
            step = step +1
        
        
        batch_rewards_to_go.extend(reward_to_go(episode_rewards)) 
        batch_log_probs.extend(episode_log_probs)
        batch_expected_rewards.extend(episode_expected_rewards)

        episode_reward = torch.sum(torch.as_tensor(episode_rewards))
        avg_reward = ((avg_reward * episode) + episode_reward.item()) / (episode+1)

        writer.add_scalar("Reward/Avg", avg_reward, (epoch *  episodes_per_epoch) + episode)
        writer.add_scalar("ActionProbDiff/Avg", action_prob_diff_avg,  (epoch *  episodes_per_epoch) + episode)

    # train the policy, and value approximator
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    batch_loss = policy_loss(torch.stack(batch_log_probs), torch.stack(batch_rewards_to_go) - torch.squeeze(torch.stack(batch_expected_rewards)))
    value_loss_ = value_loss(torch.stack(batch_rewards_to_go), torch.squeeze(torch.stack(batch_expected_rewards)))
    batch_loss.backward(retain_graph=True)
    value_loss_.backward()
    policy_optimizer.step()
    value_optimizer.step()
    print("epoch ", epoch, ", loss" , batch_loss,  ", average reward ", avg_reward, ", value_loss ", value_loss_,  "\n\n")

torch.save(policy.state_dict(), "./saved-models/vpg.pth")



   
    
        
        


