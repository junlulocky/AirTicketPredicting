# system library
import operator
import json
import os.path
import numpy as np

# third-party library
import numpy as np
import matplotlib.pyplot as plt

# user-library
import qlearn
from utils import load_data
from utils import log

routes = ["BCN_BUD",  # route 1
      "BUD_BCN",  # route 2
      "CRL_OTP",  # route 3
      "MLH_SKP",  # route 4
      "MMX_SKP",  # route 5
      "OTP_CRL",  # route 6
      "SKP_MLH",  # route 7
      "SKP_MMX"]  # route 8

# random price list
randomPrices_train = [68.4391315136,
                     67.4260645161,
                     93.2808545727,
                     77.4751720047,
                     75.0340018399,
                     73.9964736451,
                     105.280932384,
                     97.1720369004]
randomPrices_test = [55.4820634921,
                          57.8067301587,
                          23.152037037,
                          33.3727319588,
                          35.3032044199,
                          41.1180555556,
                          56.3433402062,
                          60.2546519337]
# minimum price list
minPrices_train = [44.4344444444,
                       38.9605925926,
                       68.6566666667,
                       49.6566666667,
                       48.2691891892,
                       47.0833333333,
                       68.982,
                       63.1279459459]
minPrices_test = [32.370952381,
                       29.3775238095,
                       11.3788888889,
                       16.5284615385,
                       18.6184615385,
                       14.6111111111,
                       21.5127692308,
                       25.8050769231]

# maximum price list
maxPrices_train = [115.915925926,
                        126.782814815,
                        144.212222222,
                        129.656666667,
                        141.252972973,
                        149.972222222,
                        174.402,
                        160.91172973
                        ]

maxPrices_test = [126.656666667,
                       168.95847619,
                       93.6011111111,
                       90.5669230769,
                       101.233846154,
                       198.361111111,
                       154.505076923,
                       208.020461538]

class QLearningAgent():
    def __init__(self, datas):
        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: current price
        self.datas = datas # datas have same departure date
        self.actions = 2  # action=0 for buy; action=1 for wait.

        states = np.unique(self.datas[:,9])
        self.maxStates = max(states) # states range from 0 to maxStates(totally maxStates+1)
        self.qlearning = qlearn.QLearn(self.actions, self.maxStates)

        # initialize the action = buy
        for i in range(self.datas.shape[0]):
            state = self.datas[i, 9]
            reward = -1 * self.datas[i, 10]
            self.qlearning.updateQForState(state, 0, reward)

        # initialize the action = wait
        for state in range(int(self.maxStates+1)):
            reward = -1 * self.getMinimumFuturePrice(state)
            self.qlearning.updateQForState(state, 1, reward)



        """
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
        """
        # for state = 0, the action = wait means nothing
        self.qlearning.updateQForState(0, 1, -1 * self.getPrice(0))

    def getPrice(self, state):
        """
        Given the agent(same flight number and same departhre date), and the state, return the price
        :param state: input state
        :return: price corresponding to the state
        """
        price = float("inf")
        for i in range(self.datas.shape[0]):
            if self.datas[i, 9] == state:
                price = self.datas[i, 10]
        return price

    """
    def getPrice(self, state):
        price = float("inf")
        for data in self.datas:
            if data["State"] == state:
                price = float( filter( lambda x: x in '0123456789.', data["MinimumPrice"]) )
        return price
    """

    def getMinimumFuturePrice(self, state):
        """
        Get the minimum price in the future after the state to update the QValue matrix
        :param state: input state
        :return: minimum price after the state
        """
        price = float('+inf')
        minPrice = float('+inf')
        for i in range(self.datas.shape[0]):
            if state == self.datas[i, 9]:
                price = self.datas[i, 10] # get the current state price
                minPrice = self.datas[i, 10] # initialize the minimum price
            if state >= self.datas[i, 9] and self.datas[i, 10]<=minPrice:
                minPrice = self.datas[i, 10]
        if state == 0:
            return price

        return minPrice

    """
    def getMinimumFuturePrice(self, state):
        minPrice = self.getPrice(state-1)
        for i in range(state):
            minPrice = min(minPrice, self.getPrice(i))

        return minPrice
    """


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

def getDataFromRegression():
    """
    Get the Input datas from the regression data
    :return: input datas for qlearning
    """
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: current price
    X_train = np.load('inputReg/X_train.npy')
    price_train = X_train[:,12]
    price_train = price_train.reshape((price_train.shape[0], 1))

    X_test = np.load('inputReg/X_test.npy')
    price_test = X_test[:,12]
    price_test = price_test.reshape((price_test.shape[0], 1))

    # get the input data for qlearning
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: current price
    qdata_train = np.concatenate((X_train[:,0:10], price_train), axis=1)
    np.save('inputQLearning/qdata_train', qdata_train)
    qdata_test = np.concatenate((X_test[:,0:10], price_test), axis=1)
    np.save('inputQLearning/qdata_test', qdata_test)


"""
def mainLearnOneRoute(filePrefix="BCN_BUD", dataset="large data set"):

    print "##########  Begin Learning  ##########"
    # get the total departure date length in this route
    departureLen = load_data.get_departure_len(filePrefix, dataset)

    # keep the final Q Values for each departure date
    qvalues = []
    routeMaxStates = 0 # keep track of the maxStates for the route, finally states range from (0, routeMaxStates+1)
    for index in range(departureLen):
        datas = load_data.load_data_with_departureIndex(index, filePrefix, dataset)
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
"""

def mainTrainOneRoute(filePrefix):
    """
    learn q matrix for one route
    :param filePrefix: flight prefix
    :return: the chosen state for the route
    """
    print "##########  Begin Learning  ##########"
    # load qlearning datas
    # get the input data for qlearning
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: current price
    qdatas = np.load('inputQLearning/qdata_train.npy')
    qdatas = qdatas[np.where(qdatas[:,8]>=20)[0], :]

    # choose one route datas
    flightNum = routes.index(filePrefix)
    qdatas = qdatas[np.where(qdatas[:, flightNum]==1)[0], :]

    # get all the departure dates
    departureDates = np.unique(qdatas[:,8])

    # keep track of the maxStates for the route, finally states range from (0, routeMaxStates+1)
    routeMaxStates = np.amax(qdatas[:,9])

    # keep the final Q Values for each departure date
    qvalues = []
    for departureDate in departureDates:
        datas = qdatas[np.where(qdatas[:,8]==departureDate)[0], :]
        agent = QLearningAgent(datas)
        maxStates = agent.qlearning.maxStates
        qvalues.append(agent.qlearning)

    # perform average step
    QValueTotal = {}
    QValueCount = {}
    avgQValue = {}
    for state in range(int(routeMaxStates)):
        QValueTotal[state] = 0
        QValueCount[state] = 0
        for qvalue in qvalues:
            if qvalue.getQ(state,0) != float("inf") and qvalue.getQ(state,0) != float("-inf"):
                QValueTotal[state] += qvalue.getQ(state,0)
                QValueCount[state] += 1
        avgQValue[state] = QValueTotal[state]/QValueCount[state]

    print "Q Values(state, totalValue): {}".format(QValueTotal)
    print "Q Values(state, count): {}".format(QValueCount)
    print "Q Values(state, avgValue): {}".format(avgQValue)

    # get the maximum Q Value state, and corresponding reward
    chosenState = max(avgQValue.iteritems(), key=operator.itemgetter(1))[0]
    chosenReward = max(avgQValue.iteritems(), key=operator.itemgetter(1))[1]
    print "Chosen State: {}".format(chosenState)
    print "Chosen State Reward:, {}".format(chosenReward)

    print "##########  End Learning  ##########"
    return chosenState

"""
def evaluateOneRoute(chosenState, filePrefix="BCN_BUD", dataset="large data set"):
    print "##########  Begin Evaluating  ##########"
    # get the total departure date length in this route
    departureLen = load_data.get_departure_len(filePrefix, dataset)

    # For every departure date, get the minimum price, and chosen price
    minimumPrice = {}
    chosenPrice = {}

    for index in range(departureLen):
        # get the dataset with same departure date
        datas = load_data.load_data_with_departureIndex(index, filePrefix, dataset)

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
"""
def evaluateOneRoute(chosenState, filePrefix="BCN_BUD", isTrain=False):
    print "##########  Begin Evaluating  ##########"
    # load qlearning datas
    # get the input data for qlearning
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: current price
    if isTrain:
        qdatas = np.load('inputQLearning/qdata_train.npy')
        qdatas = qdatas[np.where(qdatas[:,8]>=20)[0], :]
    else:

        qdatas = np.load('inputQLearning/qdata_test.npy')
        qdatas = qdatas[np.where(qdatas[:,8]>=20)[0], :]
        """
        qdatas1 = np.load('inputQLearning/qdata_train.npy')
        qdatas1 = qdatas1[np.where(qdatas1[:,8]>=20)[0], :]
        qdatas = np.load('inputQLearning/qdata_test.npy')
        qdatas = qdatas[np.where(qdatas[:,8]>=20)[0], :]
        qdatas = np.concatenate((qdatas, qdatas1), axis=0)
        """


    # choose one route datas
    flightNum = routes.index(filePrefix)
    qdatas = qdatas[np.where(qdatas[:, flightNum]==1)[0], :]

    # get the departure dates
    departureDates = np.unique(qdatas[:, 8])

    # get the chosen state prices
    prices = np.empty(shape=(0, qdatas.shape[1]))
    lastBuyState = 0 # if no chosen state data, then use the last buy state
    for departureDate in departureDates:
        datas = qdatas[np.where(qdatas[:,8]==departureDate)[0], :]
        data = datas[np.where(datas[:, 9]==chosenState)[0], :]
        if data.shape[0] == 0:
            data = datas[np.where(datas[:, 9]==lastBuyState)[0], :]
        prices = np.concatenate((prices, data), axis=0)
    prices = prices[:, 10]


    print "Counts: {}; Chosen Prices:{}".format(len(prices), prices)
    avgPrice = np.mean(prices)

    if isTrain:
        print "TRAIN:"
        print "minimumPrice: {}".format(minPrices_train[flightNum])
        print "maximumPrice: {}".format(maxPrices_train[flightNum])
        print "randomPrice: {}".format(randomPrices_train[flightNum])
        print "avgPredPrice: {}".format(avgPrice)

        performance = (randomPrices_train[flightNum] - avgPrice) / randomPrices_train[flightNum] * 100
        print "Performance: {}%".format(round(performance,2))
        maxPerformance = (randomPrices_train[flightNum] - minPrices_train[flightNum]) / randomPrices_train[flightNum] * 100
        print "Max Perfor: {}%".format(round(maxPerformance,2))
        normalizedPefor = performance / maxPerformance * 100
        print "Normalized perfor: {}%".format(round(normalizedPefor,2))
    else:
        print "TEST:"
        print "minimumPrice: {}".format(minPrices_test[flightNum])
        print "maximumPrice: {}".format(maxPrices_test[flightNum])
        print "randomPrice: {}".format(randomPrices_test[flightNum])
        print "avgPredPrice: {}".format(avgPrice)

        performance = (randomPrices_test[flightNum] - avgPrice) / randomPrices_test[flightNum] * 100
        print "Performance: {}%".format(round(performance,2))
        maxPerformance = (randomPrices_test[flightNum] - minPrices_test[flightNum]) / randomPrices_test[flightNum] * 100
        print "Max Perfor: {}%".format(round(maxPerformance,2))
        normalizedPefor = performance / maxPerformance * 100
        print "Normalized perfor: {}%".format(round(normalizedPefor,2))

    print "##########  End Evaluating  ##########"

    return (performance, normalizedPefor)

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

def MainTestOneRoute(filePrefix, isTrain):
    """
    Test one Flight Number(i.e. one route), from specific dataset(large or small)
    :param filePrefix: route prefix
    :param dataset: large or small dataset
    :return: performance
    """
    chosenState = mainTrainOneRoute(filePrefix)
    [perfor, normaPefor] = evaluateOneRoute(chosenState, filePrefix, isTrain)

    return [perfor, normaPefor]



if __name__ == "__main__":

    #testOneRoute(routes[7], "large data set")
    #print getPerformanceFromJson(routes[0])
    #getDataFromRegression()
    performance = 0
    normalizedPerformance = 0
    isTrain = 0 # 1 for train; 0 for test
    normPerforms = []
    for i in range(8):
        print "Route: {}".format(i)
        [perfor, normaPefor] = MainTestOneRoute(routes[i], isTrain)
        normPerforms.append(normaPefor)
        performance += perfor
        normalizedPerformance += normaPefor

    performance = round(performance/8, 2)
    normalizedPerformance = round(normalizedPerformance/8, 2)

    print "\nAverage Performance: {}%".format(performance)
    print "Average Normalized Performance: {}%".format(normalizedPerformance)
    print "Normalized Performance Variance: {}".format(np.var(normPerforms))





