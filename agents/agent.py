from torch.distributions import Categorical
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from PIL import Image


def preprocess_image(observation):
    """
    This function is used is used to done the preprocessing to the image.
    """
    reduced_obs = observation[::2, ::2]
    background = [33, 1, 36]
    black_white_obs = np.zeros((reduced_obs.shape[0], reduced_obs.shape[1], 1))

    for i in range(reduced_obs.shape[0]):
        for j in range(reduced_obs.shape[1]):
            comparison = reduced_obs[i, j] != background
            if comparison.all():
                black_white_obs[i, j] = 1

    black_white_obs = black_white_obs.transpose((2, 0, 1))
    return torch.from_numpy(black_white_obs.astype(np.float32)).unsqueeze(0)


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
    def __init__(self, input_image_shape, action_space):
        super(Policy, self).__init__()
        self.eps_clip = 0.1

        # First part of the neural network are the convolutional layers.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.ReLU()
        )

        # Calculation of input size -> w-m+1 (for each convolution)
        # Second part are the linear layers.
        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 96 * 96, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, observation):
        """
        In this function the observation is processed in the neural network
        """
        logits = self.conv_layers(observation)
        # print("In forward 1 ", torch.sum(torch.isnan(logits)).item(), len(logits))
        # Convolutional layer is reshaped to fit linear layers.
        logits = logits.view(observation.shape[0], -1)
        # print("In forward 2 ", torch.sum(torch.isnan(logits)).item(), len(logits))
        logits = self.linear_layers(logits)
        # print("In forward 3 ", torch.sum(torch.isnan(logits)).item(), len(logits))
        return logits


class Agent(object):
    def __init__(self, policy, env):
        self.env = env
        self.eps_clip = 0.1
        self.name = "Rafa Nadal"

        # TODO: Change possibly to GPU
        self.train_device = "cpu"

        # Set the policy which is going to be used.
        self.policy = policy.to(self.train_device)

        # Definition of optimizer and gamma parameter.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5E-4)
        self.gamma = 0.98

        # Save the intermediate and middle action probs and rewards
        self.observations = []
        self.actions = []
        self.action_probs = []
        self.rewards = []

    # TODO: Implement model loading.
    def load_model(self, model_path):
        """
        Function that loads the model file.
        """
        self.policy.load_state_dict(torch.load(model_path))

    # TODO: Implement reset function.
    def reset(self):
        """
        Method that resets the state of the agent
        """
        self.observations, self.actions, self.action_probs, self.rewards = [], [], [], []

        for layer in self.policy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
            # if torch.sum(torch.isnan(logits)).item() > 0:
            # print("logits is nan ", logits)
            distribution = Categorical(logits=logits)
            # print ("In get_action and dist = ", distribution)
            action = distribution.sample()
            # action_prob = distribution.probs[0, action]
            action_prob = distribution.log_prob(action)

        return action, action_prob

    # Implement the update of the policy.
    def update_policy(self):
        if (self.rewards[-1].item() == 10):
            print("win!")

        observations = torch.stack(self.observations, dim=0).to(self.train_device).squeeze(1)
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
        r = torch.sum(logits * ts, dim=1) / action_probs
        # print(r)
        loss1 = r * discounted_rewards
        loss2 = torch.clamp(action_probs, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

    def store_outcome(self, observation, action, action_prob, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
