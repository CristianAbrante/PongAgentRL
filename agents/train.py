import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from agent import Agent, Policy
import wimblepong


# Policy training function
def train(train_run_id, train_episodes, log_result=True):
    # Create a Gym environment
    env = gym.make("WimblepongVisualMultiplayer-v0")

    # Get dimensionalities of actions and observations
    action_space_dim = 1
    observation_space_dim = env.observation_space.shape

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action.detach().numpy())

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(observation, action_probabilities, reward)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if log_result:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Update agent policy
        agent.update_policy(episode_number)

    # Training is finished - plot rewards
    if log_result:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("AC reward history (episodic)")
        plt.savefig("plots/task-1.png")
        plt.show()
        print("Training finished.")

    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id] * len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Episodic AC"] * len(reward_history),
                         "reward": reward_history})

    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % ("WimblepongVisualMultiplayer-v0", train_run_id))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", default=True, action='store_true', help="Render test")
    args = parser.parse_args()

    # TODO: Add the option to test the result possible
    train(0, train_episodes=args.train_episodes, log_result=args.render_test)
