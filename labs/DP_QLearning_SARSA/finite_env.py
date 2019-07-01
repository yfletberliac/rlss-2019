from abc import ABC, abstractmethod
import numpy as np


class FiniteEnv(ABC):
    """
    Base class for a finite MDP.

    Args:
        states      (list): List of legnth S containing the indexes of the states, e.g. [0,1,2]
        action_sets (list): List containing the actions available in each state, e.g. [[0,1], [2,3]],
                            action_sets[i][j] returns the index of the j-th available action in state i
        P       (np.array): Array of shape (Ns, Na, Ns) containing the transition probabilities,
                            P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a). Na is the total number of actions.
        gamma      (float): discount factor


    Attributes:
        Ns   (int): Number of states
        Na   (int): Number of actions
        actions (list): list containing all possible actions = [0, 1, ..., Na-1]

    """
    def __init__(self, states, action_sets, P, gamma):

        self.states = states
        self.action_sets = action_sets
        self.actions = list(set().union(*action_sets))
        self.Ns = len(states)
        self.Na = len(self.actions)
        self.P = P

        self.state = 0  # initial state
        self.gamma = gamma
        self.reset()
        super().__init__()

    def available_actions(self, state=None):
        """
        Return all actions available in a given state.
        """
        if state is not None:
            return self.action_sets[state]
        return self.action_sets[self.state]

    @abstractmethod
    def reset(self):
        """
        Reset the environment to a default state.

        Returns:
            state (object)
        """
        pass

    @abstractmethod
    def reward_func(self, state, action, next_state):
        """
        Args:
            state      (int): current state
            action     (int): current action
            next_state (int): next state

        Returns:
            reward (float)
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Args:
            action (int): index of the action to take

        Returns:
            observation (object)
            reward      (float)
            done        (bool)
            info        (dict)
        """
        pass

    @abstractmethod
    def sample_transition(self, s, a):
        """
        Sample a transition s' from P(s'|s,a).

        Args:
            s (int): index of state
            a (int): index of action

        Returns:
            ss (int): index of next state
        """
        pass
