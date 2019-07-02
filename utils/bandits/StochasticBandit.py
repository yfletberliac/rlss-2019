import numpy as np
import Arms as arms

class MAB:
    def __init__(self,arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def generateReward(self,arm):
        return self.arms[arm].sample()

## some functions that create specific MABs

def BernoulliBandit(means):
    """define a Bernoulli MAB from a vector of means"""
    return MAB([arms.Bernoulli(p) for p in means])

def RandomBernoulliBandit(Delta,K):
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand()*(1.-Delta)
    secondmaxMean= maxMean-Delta
    means = secondmaxMean*np.random.random(K)
    bestarm = np.random.randint(0,K)
    secondbestarm = np.random.randint(0,K)
    while (secondbestarm==bestarm):
        secondbestarm = np.random.randint(0,K)
    means[bestarm]=maxMean
    means[secondbestarm]=secondmaxMean
    return BernoulliBandit(means)
