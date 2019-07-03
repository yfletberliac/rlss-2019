from scipy.stats.distributions import beta
import matplotlib.pyplot as plt
import pandas as pd

from reco_env import RecoEnv

import time


def train_eval_online(env, num_users, agent, mode='train'):
    """
    Trains or evaluates the agent in the environment by sampling a given number of users

    :param env: recommendation environment
    :param num_users: number of users to sample from environment
    :param agent: agent
    :param mode: train or eval
    :return: tuple of agent class, 50% quantile of CTR, 2.5% quantile of CTR, 97.5% quantile of CTR, execution time
    """
    num_clicks, num_displays = 0, 0

    start = time.time()
    for _ in range(num_users):
        env.reset()
        observation, _, _, _ = env.step(None)
        reward, done = None, False
        while not done:
            # choose action
            action = agent.act(observation)
            # execute action in the environment
            next_observation, reward, done, info = env.step(action['a'])
            # train on the feedback
            if mode == 'train':
                agent.train(observation, action['a'], reward, done)
            # compute click through rate
            num_clicks += 1 if reward == 1 and reward is not None else 0
            num_displays += 1 if reward == 0 and reward is not None else 0
            # update observation
            observation = next_observation

    end = time.time()
    result = (
        type(agent).__name__,
        beta.ppf(0.500, num_clicks + 1, num_displays + 1),
        beta.ppf(0.025, num_clicks + 1, num_displays + 1),
        beta.ppf(0.975, num_clicks + 1, num_displays + 1),
        end - start
    )
    return result


col_names = ['Agent', '0.500', '0.025', '0.975', 'Time']

def plot_ctr(eval_ctr):
    """
    Plots agents average CTR and confidence interval

    :param eval_ctr: dataframe containing agent class, 50% quantile of CTR, 2.5% quantile of CTR, 97.5% quantile of CTR
    """
    fig, ax = plt.subplots()
    ax.set_title('CTR Estimate')
    plt.errorbar(eval_ctr['Agent'],
                 eval_ctr['0.500'],
                 yerr=(eval_ctr['0.500'] - eval_ctr['0.025'],
                       eval_ctr['0.975'] - eval_ctr['0.500']),
                 fmt='o',
                 capsize=4)
    plt.xticks(eval_ctr['Agent'], eval_ctr['Agent'], rotation='vertical')
    plt.ylabel('CTR')
    plt.show()


def train_eval_agent(agent, config, num_train_users, num_eval_users):
    """
    Trains and evaluates one agent on a given number of users

    :param agent: agent to train and evaluate
    :param config: configuration of recommendation environment
    :param num_train_users: number of train users
    :param num_eval_users: number of evaluation users
    :return:
    """
    env = RecoEnv(config)
    train_eval_online(env, num_train_users, agent, 'train')
    stats = train_eval_online(env, num_eval_users, agent, 'eval')
    return stats


def train_eval_agents(agents, config, num_train_users, num_eval_users):
    """
    Trains and evaluates a list of agents on a given number of users

    :param agents: list of agents to evaluate
    :param config: configuration of recommendation environment
    :param num_train_users: number of train users
    :param num_eval_users: number of evaluation users
    :return dataframe containing agent class, 50% quantile of CTR, 2.5% quantile of CTR, 97.5% quantile of CTR, execution time
    """
    stats = [train_eval_agent(agent, config, num_train_users, num_eval_users) for agent in agents]
    df = pd.DataFrame(stats, columns=col_names)
    return df
