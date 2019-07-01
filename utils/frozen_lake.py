import numpy as np
import os
from finite_env import FiniteEnv


class MDP(FiniteEnv):
    """
    Enviroment with 3 states and 2 actions per state that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        gamma (float): discount factor
        seed    (int): Random number generator seed

    """

    def __init__(self, P, bad_states=[], gamma=0.99, seed=42):
        # Set seed
        self.RS = np.random.RandomState(seed)

        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)
 
        P = P
        Ns, Na, _ = P.shape
        
        self.Ns = Ns
        self.Na = Na
        self.bad_states = set(bad_states)

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
        
        if self.state in self.bad_states or self.state == self.Ns-1:
            done = True
        else:
            done = False
            
        info = {}
        self.state = next_state

        observation = next_state
        return observation, reward, done, info

    def sample_transition(self, s, a):
        prob = self.P[s,a,:]
        s_ = self.RS.choice(self.states, p = prob)
        return s_
    
    def render(self):        
        
        env_to_print = "S"

        
        for state in range(1,self.Ns-1) :
            if state % 4 == 0:
                env_to_print += "\n"
            
            if state in self.bad_states:
                env_to_print += "H"
            else:
                env_to_print += "F"
                    
        env_to_print += "G"
        
        print("(S: starting point, safe) (F: frozen surface, safe) (H: hole, fall to your doom) (G: goal, where the frisbee is located)")
        print("=================")
        print(env_to_print)
        print("=================")
        print("Current state", self.state)
        
        
class FrozenLake(MDP):
    def __init__(self, gamma=0.99, deterministic=False, data_path="./data"):
        if deterministic:
            P = np.load(os.path.join(data_path, "frozen_lake_deterministic_transition.npy"))
        else:
            P = np.load(os.path.join(data_path, "frozen_lake_stochastic_transition.npy"))
        bad_states = [5, 7, 11, 12]
        super().__init__(P=P, bad_states=bad_states, gamma=gamma)