from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch import tensor
from torch.distributions import Normal
import numpy as np


#new imports
import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
##
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(nn.Module):

    def __init__(self):
            super(Policy, self).__init__()

            self.gamma = 0.99
            self.eps_clip = 0.1

            self.layers = nn.Sequential(
                nn.Linear(22500, 512), nn.ReLU(),
                nn.Linear(512, 2),
            )

    def state_to_tensor(self, I):
        #print(" state to tensor ", I[0])
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector. See Karpathy's post: http://karpathy.github.io/2016/05/31/rl/ """
        #print("image is ", type(I[0]))

        if I is None:
            return torch.zeros(1, 22500)
        img = I[0]
        img = img[35:185]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        img = img[::2, ::2, ::]  # downsample by factor of 2.
        img[img == 144] = 0  # erase background (background type 1)
        img[img == 109] = 0  # erase background (background type 2)
        img[img != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        #print("Shape = ", torch.from_numpy(img.astype(np.float32).ravel()).unsqueeze(0).shape)
        return torch.from_numpy(img.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, x, prev_x):
        return self.state_to_tensor(x) - self.state_to_tensor(prev_x)

    def convert_action(self, action):
        return action + 2

    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            #print(" None action ")
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob
        '''
        # policy gradient (REINFORCE)
        logits = self.layers(d_obs)
        loss = F.cross_entropy(logits, action, reduction='none') * advantage
        return loss.mean()
        '''

        # PPO
        #print("PPO")
        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss

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

    def get_action(self, observation, evaluation=True):
        print("Get_action observation ",(observation).shape)
        #obs = observation[0]
        #print(type(obs))
        #obs = obs.reshape(200*200*3)
        #print("obs.shape ",obs.shape)
        x = torch.from_numpy(observation).float().to(self.train_device)
        print("x.shape ",x.shape)
        # Pass state x through the policy network (T1)
        #print("x is ",x, type(x))
        print("-------------------------------------")
        #y = x.resize_((200*200*3))
        #print("y is ",y.shape)
        print("get_action and x = ", x.shape)
        aprob = self.policy.forward(x)
        print(" aprob is", aprob, type(aprob))
        # TODO: Not sure how to follow procedure here because output should
        # be either 0/1 (up or down)

        # Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = aprob.mean()
        else:
            action = aprob.sample()

        # DONE: Calculate the log probability of the action (T1)
        act_log_prob = aprob.log_prob(action)
        print("Action ",action)
        action = 2 if np.random.uniform() < aprob else 3
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
