import gym
import torch
import wimblepong
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('WimblepongVisualSimpleAI-v0')
train_episodes = 5

# implement this part in the agent
# opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

x_train, y_train, rewards = [], [], []
reward_sum = 0
episode_nb = 0

# Arrays to keep track of rewards
reward_history, timestep_history = [], []
average_reward_history = []

# The training loop is run per episode
for episode in range(train_episodes):
    reward_sum, timesteps = 0, 0
    done = False

    # the environment is reset each episode.
    observation = env.reset()

    while not done:
        # TODO: Get action from the agent agent.get_action(observation)
        # TODO: Compute the difference between current and past action.
        action, action_prob = env.MOVE_UP, 0.0
        previous_observation = observation

        # Action is performed in the environment.
        observation, reward, done, info = env.step(action)

        # TODO: Store action's outcome (so that the agent can improve its policy)
        # agent.store_outcome(previous_observation, action_prob, action, reward)

        # Store rewards
        reward_sum += reward
        timesteps += 1

    print(f"Episode {episode} finished | total reward -> {reward_sum}")

    # Keeping records for future plots
    reward_history.append(reward_sum)
    timestep_history.append(timesteps)
    avg = np.mean(reward_history[-100:] if episode > 100 else reward_history)
    average_reward_history.append(avg)

    # TODO: agent should update the policy.
    # agent.episode_finished(episode)

if __name__ == "__main__":
    print("end")
