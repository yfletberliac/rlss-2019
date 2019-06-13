import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random, os.path, math, glob, csv, base64
import gym
from gym.wrappers import Monitor
import tqdm


class A2CModel(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super().__init__()
        self.dim_observation = dim_observation
        self.n_actions = n_actions

        self.embedding = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU()
        )
        self.policy = nn.Linear(in_features=128, out_features=n_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        embedded_obs = self.embedding(x)
        policy = F.softmax(self.policy(embedded_obs), dim=-1)
        value = self.value(embedded_obs)
        return value, policy

    def value_action(self, state):
        policy, value = self.forward(state)
        action = torch.multinomial(policy, 1)
        return value, action


class A2CAgent:

    def __init__(self, gamma, env):
        self.env = env
        dim_observation, n_actions = env.observation_space.shape[0], env.action_space.n
        self.model = A2CModel(dim_observation=dim_observation, n_actions=n_actions)
        self.gamma = gamma
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0007)

    def _make_returns(self, rewards):
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = (rewards[t] + self.gamma * returns[t + 1])  # / (len(rewards) - t)
        return returns

    def optimize_model(self, n_trajectories):
        reward_trajectories = []
        value_losses = []
        policy_losses = []
        entropy_losses = []
        for i in range(n_trajectories):
            rewards_episode, logproba_episode, values_episode, entropy_episode = self.collect_trajectory()
            reward_trajectories.append(rewards_episode.sum())

            returns = self._make_returns(rewards_episode)
            returns = torch.tensor(returns, dtype=torch.float32)

            value_loss = torch.mean((values_episode - returns)**2)
            value_losses.append(value_loss)

            policy_loss = torch.mean(logproba_episode * (returns - values_episode.detach()))
            policy_losses.append(policy_loss)

            entropy_losses.append(entropy_episode)

        value_losses = torch.stack(value_losses).mean()
        print(value_losses)
        policy_losses = torch.stack(policy_losses).mean()
        entropy_losses = torch.stack(entropy_losses).mean()
        loss = 0.5 * value_losses - policy_losses - 0.0001 * entropy_losses
        # print(loss)
        self.optimizer.zero_grad()
        # Compute the gradient
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()

        reward_trajectories = np.array(reward_trajectories)
        return loss.detach().numpy(), reward_trajectories.mean(), reward_trajectories.std(), reward_trajectories.min(), reward_trajectories.max()

    def collect_trajectory(self):
        # New episode
        observation = self.env.reset()
        rewards_episode = []
        values_episode = []
        logproba_episode = []
        entropy_episode = []
        observation = torch.tensor(observation, dtype=torch.float)
        done = False

        while not done:
            value, policy = self.model(observation)
            action = torch.multinomial(policy, 1)
            values_episode.append(value)
            entropy_episode.append(- torch.sum(policy * torch.log(policy)))
            logproba_episode.append(torch.log(policy)[action])

            # Interaction with the environment
            observation, reward, done, info = self.env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            rewards_episode.append(reward)

        rewards_episode = np.array(rewards_episode)
        logproba_episode = torch.cat(logproba_episode)
        values_episode = torch.cat(values_episode)
        entropy_episode = torch.sum(torch.stack(entropy_episode))

        return rewards_episode, logproba_episode, values_episode, entropy_episode

    def train(self, n_trajectories, n_update):
        mean_rewards = np.zeros(n_update)
        losses = np.zeros(n_update)
        for episode in tqdm.trange(n_update):
            loss, mean_reward, std_reward, min_reward, max_reward = self.optimize_model(n_trajectories)
            mean_rewards[episode] = mean_reward
            losses[episode] = loss
            # print("Episode {}".format(episode + 1))
            # print("Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
            #       .format(mean_reward, std_reward, min_reward, max_reward))
        plt.plot(mean_rewards)
        plt.show()
        plt.plot(losses)
        plt.show()

    def evaluate(self, n_trajectories):
        reward_trajectories = np.zeros(n_trajectories)
        for i in range(n_trajectories):
            # New episode
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode = 0
            done = False

            while not done:
                env.render()
                action = self.model.value_action(observation)[1]
                observation, reward, done, info = self.env.step(int(action))
                observation = torch.tensor(observation, dtype=torch.float)
                reward_episode += reward

            reward_trajectories[i] = reward_episode
        env.close()
        print("Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
              .format(reward_trajectories.mean(), reward_trajectories.std(),
                      reward_trajectories.min(), reward_trajectories.max()))


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    env.seed(100)
    agent = A2CAgent(0.9, env)

    agent.train(1, 1000)
