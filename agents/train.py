import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from agents.agent import Agent, Policy
import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
import wimblepong

env = gym.make("WimblepongVisualMultiplayer-v0")
env.reset()
policy = Policy()
#("Param = ",policy.parameters)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

def preprocessing(image):
    #print("I am preprocessing ",image[0].shape)
    im = image[0]
    img = im[::2,::2,::] #downsampling by a factor of 2
    #print("img shape ",img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])
    #print("prep ")
    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    avg = (R + G + B)
    grayImage = img

    for i in range(3):
        grayImage[:, :, i] = avg
    #print(grayImage.astype(np.float).ravel().shape)
    return grayImage.astype(np.float).ravel()#we return a gray image that we have downsampled

# Policy training function
#print("I am training")


    ##main loop
reward_sum_running_avg = None
d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
for it in range(5000):

        for ep in range(10):
            obs, prev_obs = env.reset(), None
            for t in range(190000):
                # env.render()

                d_obs = policy.pre_process(obs, prev_obs)
                with torch.no_grad():
                    action, action_prob = policy(d_obs)

                prev_obs = obs
                #print(" action is ",action)
                #action = 0
                obs, reward, done, info = env.step(action)
                reward = reward[0]#multiplayer mode
               # print("reward is ", env.step(policy.convert_action(action)))
                d_obs_history.append(d_obs)
                action_history.append(action)
                action_prob_history.append(action_prob)
                reward_history.append(reward)

                if done:
                    #print(reward_history[-t])
                    reward_sum = sum(reward_history[-t:])
                    reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                    print(
                        'Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (
                        it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                    break


        # compute advantage
        R = 0
        discounted_rewards = []

        for r in reward_history[::-1]:
            if r != 0: R = 0  # scored/lost a point in pong, so reset reward sum
            R = r + policy.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        # update policy
        for _ in range(5):
            n_batch = 50##maximum is 109
            #print("History ", range(len(action_history)))
            idxs = random.sample(range(len(action_history)), n_batch)
            d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
            action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
            action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
            advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])


            ##def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False)
            #print( "loss is ", d_obs_batch,action_batch, action_prob_batch, advantage_batch)
            #print("end loss ")
            opt.zero_grad()
            #policy1 = Policy()
            loss = policy.forward(d_obs_batch,action_batch, action_prob_batch, advantage_batch)
            #print(loss)
            loss.backward()
            opt.step()

        if it % 5 == 0:
            torch.save(policy.state_dict(), 'params.ckpt')
            plt.plot(reward_history)
            plt.legend(["Reward", "Episodes"])
            plt.title("Reward history")
            #plt.show()
            plt.savefig("rewards.png")
            print("Training finished.")





plt.plot(reward_history)
plt.legend(["Reward", "100-episode average"])
plt.title("Reward history")
plt.show()
print("Training finished.")

env.close()





    # Create a Gym environment


    ##end main loop
    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", default=True, action='store_true', help="Render test")
    args = parser.parse_args()


    #train(0, train_episodes=args.train_episodes, log_result=args.render_test)
