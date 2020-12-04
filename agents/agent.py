from torch.distributions import Categorical
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn


def preprocess_image(observation):
    """
    This function is used is used to done the preprocessing to the image.
    """
    image = np.array(observation)
    background = [33, 1, 36]
    preprocessed_img = np.zeros((observation.shape[0], observation.shape[1], 1))

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            comparison = image[i, j] == background
            if comparison.all():
                preprocessed_img[i, j] = 0
            else:
                preprocessed_img[i, j] = 1
    return torch.from_numpy(preprocessed_img.astype(np.float32).ravel()).unsqueeze(0)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def subtract_observations(current_obs, prev_obs=None):
    if prev_obs is None:
        return preprocess_image(current_obs)
    else:
        return preprocess_image(current_obs) - preprocess_image(prev_obs)


class Policy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Policy, self).__init__()
        self.eps_clip = 0.1

        # TODO: Change structure with CNN.
        # pixels_size = image_shape[0] * image_shape[1] * image_shape[2]
        # Neural network is defined as a sequential structure.
        self.layers = nn.Sequential(
            nn.Linear(observation_space, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, observation):
        """
        In this function the observation is processed in the neural network
        """
        # print(observation)
        logits = self.layers(observation)
        return logits


class Agent(object):
    def __init__(self, policy, env):
        self.env = env
        self.eps_clip = 0.1
        self.name = "Simplest AI"

        # TODO: Change possibly to GPU
        self.train_device = "cpu"

        # Set the policy which is going to be used.
        self.policy = policy.to(self.train_device)

        # Definition of optimizer and gamma parameter.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5E-3)
        self.gamma = 0.98

        # Save the intermediate and middle action probs and rewards
        self.observations = []
        self.actions = []
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

    def get_action(self, observation, deterministic=True):
        """
        Get action from observation, observation is already in desired format
        """
        # Observation is computed.
        processed_observation = observation.to(self.train_device)

        # The output of the network is computed here
        logits = self.policy.forward(processed_observation)

        # If deterministic take the maximum logit, else, sample from categorical distribution.
        if deterministic:
            action = torch.argmax(logits[0])
            action_prob = torch.Tensor(1.0)
        else:
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            action_prob = distribution.log_prob(action)

        return action, action_prob

    # Implement the update of the policy.
    def update_policy(self, episode_number):
        observations = torch.stack(self.observations, dim=0).to(self.train_device).squeeze()
        actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        # Internal state is reset.
        self.observations, self.actions, self.action_probs, self.rewards = [], [], [], []

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, self.gamma)

        self.optimizer.zero_grad()

        # Compute the optimization based on the loss
        # Application of PPO
        vs = np.array([[1., 0., 0], [0., 1., 0.], [0., 0., 1.]])
        ts = torch.FloatTensor(vs[actions.cpu().numpy()])

        # Update of the network params using loss.
        logits = self.policy.forward(observations)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_probs

        loss1 = r * discounted_rewards
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

    def store_outcome(self, observation, action, action_prob, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
