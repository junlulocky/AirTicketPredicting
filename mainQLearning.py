# system library
import operator
import json
import os.path

# third-party library
import numpy as np
import matplotlib.pyplot as plt

# user-library
import qlearn
import load_data
import log



class QLearningAgent():
    def __init__(self, datas):
        self.datas = datas
        self.actions = 2  # action=0 for buy; action=1 for wait.


        states = []
        [states.append(data["State"]) for data in self.datas]
        self.maxStates = max(states) # states range from 0 to maxStates(totally maxStates+1)
        self.qlearning = qlearn.QLearn(self.actions, self.maxStates)

        # initialize the action = buy
        for state in range(self.maxStates+1):
            try:
                reward = -1 * self.getPrice(state)
                self.qlearning.updateQForState(state, 0, reward)
            except: # a little tricky here
                print "Exception: state {:d}, action buy".format(state)
                reward = -1 * self.getPrice(state-1)
                self.qlearning.updateQForState(state, 0, reward)


        # initialize the action = wait
        for state in range(self.maxStates+1):
            try:
                reward = -1 * self.getMinimumFuturePrice(state)
                self.qlearning.updateQForState(state, 1, reward)
            except:
                print self.getMinimumFuturePrice(66)
                print "Exception: state {:d}, action wait".format(state)
        # for state = 0, the action = wait means nothing
        self.qlearning.updateQForState(0, 1, -1 * self.getPrice(0))

    def getPrice(self, state):
        """
        Given the agent(same flight number and same departhre date), and the state, return the price
        :param state: input state
        :return: price corresponding to the state
        """
        price = float("inf")
        for data in self.datas:
            if data["State"] == state:
                price = float( filter( lambda x: x in '0123456789.', data["MinimumPrice"]) )
        return price

    def getMinimumFuturePrice(self, state):
        """
        Get the minimum price in the future after the state to update the QValue matrix
        :param state: input state
        :return: minimum price after the state
        """
        minPrice = self.getPrice(state-1)
        for i in range(state):
            minPrice = min(minPrice, self.getPrice(i))

        return minPrice

    def visualizeQValue(self):
        """
        Visualize the Buy action and Wait action reward.
        :return: NA
        """
        wait = [self.qlearning.getQ(i, 1) for i in range(self.maxStates+1)]
        buy = [self.qlearning.getQ(i, 0) for i in range(self.maxStates+1)]
        x = range(self.maxStates+1)
        line1, = plt.plot(x, wait, 'r--')
        line2, = plt.plot(x, buy, 'bo')
        plt.legend([line1, line2], ["Action wait", "Action buy"])
        plt.xlabel('States')
        plt.ylabel('Q Value')
        plt.show()

def mainLearnOneRoute(filePrefix="BCN_BUD", dataset="large data set"):

    print "##########  Begin Learning  ##########"
    # get the total departure date length in this route
    departureLen = load_data.get_departure_len(filePrefix, dataset)

    # keep the final Q Values for each departure date
    qvalues = []
    routeMaxStates = 0 # keep track of the maxStates for the route, finally states range from (0, routeMaxStates+1)
    for index in range(departureLen):
        datas = load_data.load_data_QLearning(index, filePrefix, dataset)
        agent = QLearningAgent(datas)
        maxStates = agent.qlearning.maxStates
        routeMaxStates = maxStates if maxStates > routeMaxStates else routeMaxStates
        qvalues.append(agent.qlearning)

    # perform average step
    QValueTotal = {}
    QValueCount = {}
    avgQValue = {}
    for state in range(routeMaxStates):
        QValueTotal[state] = 0
        QValueCount[state] = 0
        for qvalue in qvalues:
            if qvalue.getQ(state,0) != float("inf") and qvalue.getQ(state,0) != float("-inf"):
                QValueTotal[state] += qvalue.getQ(state,0)
                QValueCount[state] += 1
        avgQValue[state] = QValueTotal[state]/QValueCount[state]

    log.log(QValueTotal)
    log.log(QValueCount)
    log.log(avgQValue)

    # get the maximum Q Value state, and corresponding reward
    chosenState = max(avgQValue.iteritems(), key=operator.itemgetter(1))[0]
    chosenReward = max(avgQValue.iteritems(), key=operator.itemgetter(1))[1]
    log.log(chosenState)
    log.log(chosenReward)

    with open('results/data_qlearing_avgQValue_{:}.json'.format(filePrefix), 'w') as outfile:
        json.dump(avgQValue, outfile)
    with open('results/data_qlearing_chosenState_{:}.json'.format(filePrefix), 'w') as outfile:
        json.dump({"chosenState":chosenState ,"chosenReward":chosenReward}, outfile)

    print "##########  End Learning  ##########"
    return chosenState

def evaluateOneRoute(chosenState, filePrefix="BCN_BUD", dataset="large data set"):
    print "##########  Begin Evaluating  ##########"
    # get the total departure date length in this route
    departureLen = load_data.get_departure_len(filePrefix, dataset)

    # For every departure date, get the minimum price, and chosen price
    minimumPrice = {}
    chosenPrice = {}

    for index in range(departureLen):
        # get the dataset with same departure date
        datas = load_data.load_data_QLearning(index, filePrefix, dataset)

        # if already get the minimum price, then do nothing
        #if os.path.exists('data_qlearing_minimumPrice_{:}.json'.format(filePrefix)):
        minimumPrice[index] = load_data.getMinimumPrice(datas)
        # chosenPrice may contain "None", only evaluate the ones are not "None"
        chosenPrice[index] = load_data.getChosenPrice(chosenState, datas)

    with open('results/data_qlearing_minimumPrice_{:}.json'.format(filePrefix), 'w') as outfile:
        json.dump(minimumPrice, outfile)
    with open('results/data_qlearing_chosenPrice_{:}.json'.format(filePrefix), 'w') as outfile:
        json.dump(chosenPrice, outfile)

    print "##########  End Evaluating  ##########"
    log.log(minimumPrice)
    log.log(chosenPrice)

    performance = getPerformance(minimumPrice, chosenPrice, False)
    return performance

def getPerformance(minimumPrice, chosenPrice, isJson):
    """
    Given the minimum price dic, and the chosen price dic, get the performance
    :param minimumPrice: input Minimum Price
    :param chosenPrice: input chosen price
    :return: performance
    """
    totalMinimumPrice = 0
    totalChosenPrice = 0
    length = len(chosenPrice)
    if isJson: # if the result is stored in a json file, the call methods are different
        for i in range(length):
            try:
                totalChosenPrice = totalChosenPrice + chosenPrice[str(i)] \
                    if chosenPrice[str(i)]!=None else totalChosenPrice
                totalMinimumPrice = totalMinimumPrice + minimumPrice[str(i)] \
                    if chosenPrice[str(i)]!=None else totalMinimumPrice
            except:
                print i
    else:
        for i in range(length):
            totalChosenPrice = totalChosenPrice + chosenPrice[i] \
                if chosenPrice[i]!=None else totalChosenPrice
            totalMinimumPrice = totalMinimumPrice + minimumPrice[i] \
                if chosenPrice[i]!=None else totalMinimumPrice


    return  totalMinimumPrice * 1.0 / totalChosenPrice

def getPerformanceFromJson(filePrefix):
    """
    If you want to get the performance from the stored json file, use this function
    :param filePrefix: route prefix
    :return: performance
    """
    with open('results/data_qlearing_minimumPrice_{:}.json'.format(filePrefix), 'r') as infile:
        minimumPrice = json.load(infile)
    with open('results/data_qlearing_chosenPrice_{:}.json'.format(filePrefix), 'r') as infile:
        chosenPrice = json.load(infile)
    print minimumPrice
    print chosenPrice

    performance = getPerformance(minimumPrice, chosenPrice, True)
    return performance

def testOneRoute(filePrefix, dataset):
    """
    Test one Flight Number(i.e. one route), from specific dataset(large or small)
    :param filePrefix: route prefix
    :param dataset: large or small dataset
    :return: performance
    """
    chosenState = mainLearnOneRoute(filePrefix, dataset)
    performance = evaluateOneRoute(chosenState, filePrefix, dataset)
    print "Performance: {}".format(performance)
    return


if __name__ == "__main__":
    routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8
    #testOneRoute(routes[7], "large data set")





