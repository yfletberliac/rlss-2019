from finite_env import FiniteEnv
import numpy as np


class ToyEnv1(FiniteEnv):
    """
    Enviroment with 3 states and 2 actions per state that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        gamma (float): discount factor
        seed    (int): Random number generator seed

    """

    def __init__(self, gamma=0.99, seed=42):
        # Set seed
        self.RS = np.random.RandomState(seed)

        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        Ns = 3
        Na = 2
        P = np.zeros((Ns, Na, Ns))

        P[:, 0, :] = np.array([[0.25, 0.5, 0.25], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1]])
        P[:, 1, :] = np.array([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.25, 0.25, 0.5]])

        # Initialize base class
        states = np.arange(Ns).tolist()
        action_sets = [np.arange(Na).tolist()]*Ns
        super().__init__(states, action_sets, P, gamma)

    def reward_func(self, state, action, next_state):
        return 1.0 * (next_state == self.Ns - 1)

    def reset(self, s=0):
        self.state = s
        return self.state

    def step(self, action):
        next_state = self.sample_transition(self.state, action)
        reward = self.reward_func(self.state, action, next_state)
        done = False
        info = {}
        self.state = next_state

        observation = next_state
        return observation, reward, done, info

    def sample_transition(self, s, a):
        prob = self.P[s,a,:]
        s_ = self.RS.choice(self.states, p = prob)
        return s_
