from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(nn.Module):
    def __init__(self, observation_space_dim, action_space_dim):
        print(observation_space_dim)
        print(action_space_dim)
        super().__init__()
        self.state_space = observation_space_dim
        self.action_space = action_space_dim
        self.hidden = 16
        # This is where the definition of the network starts.
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=(5, 5))
        self.linear_out = torch.nn.Linear(self.hidden, action_space_dim)

        # sigma as a learnt parameter of the network
        self.sigma = torch.nn.Parameter(torch.Tensor([10.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # This is where the input is forwarded to the network
        x = self.conv1(x)
        x = F.relu(x)
        x = self.linear_out(x)

        action_mean = self.fc2_mean(x)
        sigma = torch.sqrt(self.sigma)

        # Return a normal distribution.
        action_dist = Normal(loc=action_mean, scale=sigma)

        return action_dist


class Agent(object):
    def __init__(self, policy, player_id=1):
        # TODO: Change possibly to GPU
        self.train_device = "cpu"
        # Set the policy which is going to be used.
        self.policy = policy.to(self.train_device)

        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.name = "Simplest AI"
        # TODO: Possibly define other optimizer (maybe Adam)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    # TODO: Implement model loading.
    def load_model(self):
        """
        Function that loads the model file.
        """
        return

    # TODO: Implement reset function.
    def reset(self):
        """
        Method that resets the state of the agent
        """
        return

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, observation=None, evaluation=True):
        print(observation)
        x = torch.from_numpy(observation).float().to(self.train_device)

        # Pass state x through the policy network (T1)
        aprob = self.policy.forward(x)

        # TODO: Not sure how to follow procedure here because output should
        # be either 0/1 (up or down)

        # Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = aprob.mean
        else:
            action = aprob.sample()

        # DONE: Calculate the log probability of the action (T1)
        act_log_prob = aprob.log_prob(action)

        return action, act_log_prob

    def update_policy(self, episode_number):
        # Task 2a: update sigma of the policy exponentially decreasingly.
        # self.policy.update_sigma_exponentially(episode_number + 1)

        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # TODO: Probably update baseline.
        baseline = 0
        weighted_probs = -action_probs * (discounted_rewards - baseline)

        # Compute the gradients of loss w.r.t. network parameters (T1)
        loss = torch.mean(weighted_probs)
        loss.backward()

        # Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def store_outcome(self, observation, action_prob, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
