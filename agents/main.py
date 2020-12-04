import gym
import wimblepong
import numpy as np
from agent import Agent, Policy, subtract_observations
import matplotlib.pyplot as plt
import pandas as pd
import torch

env = gym.make('WimblepongVisualSimpleAI-v0')
train_episodes = 500

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
win_rate_history = []
average_win_rate_history = []
number_of_wins = 0

# The training loop is run per episode
for episode in range(train_episodes):
    reward_sum, timesteps = 0, 0
    done = False
    has_won = False

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

        if done and reward == 10:
            number_of_wins += 1

    # Keeping records for future plots
    timestep_history.append(timesteps)

    reward_history.append(reward_sum)
    avg_reward = np.mean(reward_history[-100:] if episode > 100 else reward_history)
    average_reward_history.append(avg_reward)

    win_rate = number_of_wins / episode if episode != 0 else 0.0
    win_rate_history.append(win_rate)
    avg_win_rate = np.mean(win_rate_history[-100:] if episode > 100 else win_rate_history)
    average_win_rate_history.append(avg_win_rate)

    # Printing section.

    if episode % 5 == 0:
        print(f"Episode {episode} finished | total reward -> {np.mean(reward_history)} | win rate -> {win_rate}")
        torch.save(agent.policy.state_dict(), f"models/model_{episode}.mdl")

    agent.update_policy(episode)

plt.plot(reward_history)
plt.plot(average_reward_history)
plt.legend(["Reward", "100-episode reward average"])
plt.title("Reward history")
plt.savefig("plots/reward-history.png")
plt.show()

plt.plot(win_rate_history)
plt.plot(average_win_rate_history)
plt.legend(["Win rate", "100-episode win rate average"])
plt.title("Win rate history")
plt.savefig("plots/win-rate-history.png")
plt.show()

torch.save(agent.policy.state_dict(), "model_%s_%d.mdl")

if __name__ == "__main__":
    print("end")
