import gym
import wimblepong
import numpy as np
from agent import Agent, Policy, subtract_observations
import matplotlib.pyplot as plt
import pandas as pd
import torch

env = gym.make('WimblepongVisualSimpleAI-v0')
train_episodes = 50

# TODO: Change when using a convolutional layer
observation_space_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_space_dim = 3

# Instantiate the agent and the policy.
policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy, env)

x_train, y_train, rewards = [], [], []
reward_sum = 0
episode_nb = 0

# Arrays to keep track of rewards
reward_history, timestep_history = [], []
average_reward_history = []
win_rate = []
average_win_rate = []

# The training loop is run per episode
for episode in range(train_episodes):
    reward_sum, timesteps = 0, 0
    done = False

    # the environment is reset each episode.
    observation, previous_observation = env.reset(), None
    while not done:
        # Compute training observation with the difference of previous and current observation.
        training_observation = subtract_observations(observation, previous_observation)

        action, action_prob = agent.get_action(training_observation, deterministic=False)
        previous_observation = observation

        # Action is performed in the environment.
        observation, reward, done, info = env.step(action.detach().cpu().numpy()[0])

        # Store action's outcome (so that the agent can improve its policy)
        agent.store_outcome(training_observation, action, action_prob, reward)

        # Store rewards
        reward_sum += reward
        timesteps += 1

    print(f"Episode {episode} finished | total reward -> {reward_sum}")

    # TODO: implement win counter.

    # Keeping records for future plots
    reward_history.append(reward_sum)
    timestep_history.append(timesteps)
    avg = np.mean(reward_history[-100:] if episode > 100 else reward_history)
    average_reward_history.append(avg)

    agent.update_policy(episode)

plt.plot(reward_history)
plt.plot(average_reward_history)
plt.legend(["Reward", "100-episode reward average"])
plt.title("Reward history")
plt.savefig("plots/reward-history.png")
plt.show()

torch.save(agent.policy.state_dict(), "model_%s_%d.mdl")

if __name__ == "__main__":
    print("end")
