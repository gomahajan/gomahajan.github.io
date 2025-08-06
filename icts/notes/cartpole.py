# Libraries to help construct neural networks
# and train them.
import torch
import torch.nn as nn
import torch.optim as optim

# Library to create and interact with environments.
import gymnasium as gym

# Define the policy network
# It first applies a linear mapping from the state space to hidden layer,
# and then applies a softmax activation to obtain the action probabilities.

class LinearPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        # This has obs_dim x n_actions many parameters.
        # Useful to inutively see how much data you need to train this.
        self.linear = nn.Linear(obs_dim, n_actions)

    def forward(self, x):
        # Basically if self.linear(x) = [2, 3, -4]
        # softmax(self.linear(x)) will normalize it to  [0.4, 0.5, 0.1]
        # And if we do batch, then x is [batch_size, obs_dim]
        # self.linear(x) will be [batch_size, n_actions]
        # softmax(self.linear(x)) will be [batch_size, n_actions]
        return torch.softmax(self.linear(x), dim=-1)

# Next, we create our Cartpole environment
env = gym.make("CartPole-v1")

# Here is how the observation and action space looks like
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
print(f"S = R^{obs_dim} and A = {list(range(n_actions))}")

policy = LinearPolicy(obs_dim, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# We train for 1000 episodes
for episode in range(1000):
    # We get our starting state from the environment
    obs, _ = env.reset()

    # We need to store log probabilities and rewards for computing gradients
    log_probs = []
    rewards = []
    done = False

    # We run our current policy until the end of episode
    while not done:
        # Turn our state into a tensor (because nn uses tensors)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # Compute action probabilities under our policy
        # Sample an action.
        action_probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Execute the action in the environment, collect reward, new state
        obs, reward, terminated, truncated, _ = env.step(action.item())

        # We need log probabilities and rewards to compute gradient, so lets save that
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)

        done = terminated or truncated
    
    # We need to turn rewards into tensors before using them.
    rewards = torch.tensor(rewards, dtype=torch.float32)

    # Compute the loss to propagate the gradients
    loss = -sum(log_probs)*sum(rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, return = {sum(rewards)}")