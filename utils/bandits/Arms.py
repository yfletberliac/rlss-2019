"""different classes of arms, all of them have a sample() method which produce rewards"""

import numpy as np
from random import random
from math import sqrt,log,exp

class Bernoulli:

    def __init__(self,p):
        # create a Bernoulli arm with mean p
        self.mean = p
        self.variance = p*(1-p)

    def sample(self):
        # generate a reward from a Bernoulli arm 
        return float(random()<self.mean)


class Gaussian:

    def __init__(self,mu,var=1):
        # create a Gaussian arm with specified mean and variance
        self.mean = mu
        self.variance = var

    def sample(self):
        # generate a reward from a Gaussian arm 
        return self.mean + sqrt(self.variance)*np.random.normal()
        

class Exponential:

    def __init__(self,p):
        # create an Exponential arm with parameter p
        self.mean = 1/p
        self.variance = 1/(p*p)

    def sample(self):
        # generate a reward from an Exponential arm 
        return -(self.mean)*log(random())

class TruncatedExponential:

    def __init__(self,p,trunc):
        # create a truncated Exponential arm with parameter p
        self.p = p
        self.trunc = trunc
        self.mean = (1.-exp(-p * trunc)) / p
        self.variance=0
        
    def sample(self):
        # generate a reward from an Exponential arm 
        return min(-(1/self.p)*log(random()),self.trunc)
