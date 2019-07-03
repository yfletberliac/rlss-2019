from numpy.random.mtrand import RandomState

# Default configuration
random_args = {
    'num_products': 100,
    'random_seed': 123,
}


class Agent:
    """
    This is an abstract Agent class.
    The class defines an interface with methods those should be overwritten for a new Agent.
    """

    def __init__(self, config):
        self.config = config

    def act(self, observation):
        """An act method takes in an observation, which could either be
           `None` or an Organic_Session (see recogym/session.py) and returns
           a integer between 0 and num_products indicating which product the
           agent recommends"""
        pass

    def train(self, observation, action, reward, done=False):
        """Use this function to update your model based on observation, action,
            reward tuples"""
        pass

    def reset(self):
        pass


class RandomAgent(Agent):
    """The world's simplest agent!"""

    def __init__(self, config):
        super(RandomAgent, self).__init__(config)
        self.rng = RandomState(config.random_seed)

    def act(self, observation):
        return {
            **{
                'a': self.rng.choice(self.config.num_products),
                'ps': 1.0 / float(self.config.num_products)
            },
        }