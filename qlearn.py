import random


class QLearn:
    def __init__(self, actions, maxStates, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {} # emulate the Q-Value matrix as a dictionary

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.actions = actions
        self.maxStates = maxStates


    def getQ(self, state, action):
        return self.q.get((state, action), float("-inf"))
        # return self.q.get((state, action), 1.0)

    def updateQForState(self, state, action, reward):
        self.q[(state, action)] = reward

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None) # set default value to be None
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def learnForState(self, state1, action1, reward):
        """

        :param state1: current state
        :param action1: current action
        :param reward: immediate reward
        :param state2: future state after taking the current action; state2 = state1 - 1
        :return: no return, update the "Q matrix"
        """
        state2 = state1 - 1
        if state2 > 0:
            maxqnew = max([self.getQ(state2, 0), self.getQ(state2, 1)])
            if action1 == 0:
                self.q[(state1, action1)] = reward # reward = -price(state1)
            else:
                self.q[(state1, action1)] = maxqnew
            #self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
        else:
            self.q[(state1, 0)] = reward
            self.q[(state1, 1)] = float("-inf")

    def chooseAction(self, state):
        """
        After training, use this function to predict the ticket price.
        :param state:
        :return: the action at a specific state.
        """
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action



import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs) < n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]