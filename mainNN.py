# system library
import numpy as np
import json
from collections import Counter
from itertools import groupby

# user-library
import load_data
import util
import log

# third-party library
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.utils import shuffle

routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8


def load(dataset="large data set"):
    # Construct the input data
    d = 12
    X_train = np.empty(shape=(0, d))
    y_train = np.empty(shape=(0,1))
    y_train_price = np.empty(shape=(0,1))
    X_test = np.empty(shape=(0,d))
    y_test = np.empty(shape=(0,1))
    y_test_price = np.empty(shape=(0,1))

    for filePrefix in routes:
        datas = load_data.load_data_with_prefix_and_dataset(filePrefix, dataset)
        for data in datas:
            print "Construct route {}, State {}, departureDate {}...".format(filePrefix, data["State"], data["Date"])
            x_i = []
            # feature 1: flight number -> dummy variables
            for i in range(len(routes)):
                """
                !!!need to change!
                """
                if i == routes.index(filePrefix):
                    x_i.append(1)
                else:
                    x_i.append(0)

            # feature 2: departure date interval from "20151109", because the first observed date is 20151109
            departureDate = data["Date"]
            """
            !!!maybe need to change the first observed date
            """
            departureDateGap = util.days_between(departureDate, "20151109")
            x_i.append(departureDateGap)

            # feature 3: observed days before departure date
            state = data["State"]
            x_i.append(state)

            # feature 4: minimum price before the observed date
            minimumPreviousPrice = getMinimumPreviousPrice(data["Date"], state, datas)
            x_i.append(minimumPreviousPrice)

            # feature 5: maximum price before the observed date
            maximumPreviousPrice = getMaximumPreviousPrice(data["Date"], state, datas)
            x_i.append(maximumPreviousPrice)

            # output
            y_i = [0]
            specificDatas = []
            specificDatas = [data2 for data2 in datas if data2["Date"]==departureDate]
            optimalState = load_data.getOptimalState(specificDatas)
            if data["State"] == optimalState:
                y_i = [1]

            # keep price info
            y_price = [data["MinimumPrice"]]

            if int(departureDate) < 20160115: # choose date before "20160201" as training data
                X_train = np.concatenate((X_train, [x_i]), axis=0)
                y_train = np.concatenate((y_train, [y_i]), axis=0)
                y_train_price = np.concatenate((y_train_price, [y_price]), axis=0)
            elif int(departureDate) < 20160220: # choose date before "20160220" as test data
                X_test = np.concatenate((X_test, [x_i]), axis=0)
                y_test = np.concatenate((y_test, [y_i]), axis=0)
                y_test_price = np.concatenate((y_test_price, [y_price]), axis=0)
            else:
                pass

    np.save('inputNN/X_train', X_train)
    np.save('inputNN/y_train', y_train)
    np.save('inputNN/y_train_price', y_train_price)
    np.save('inputNN/X_test', X_test)
    np.save('inputNN/y_test', y_test)
    np.save('inputNN/y_test_price', y_test_price)

    return X_train, y_train, X_test, y_test



def getMinimumPreviousPrice(departureDate, state, datas):
    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    minimumPreviousPrice = util.getPrice(specificDatas[0]["MinimumPrice"])
    for data in specificDatas:
        if util.getPrice(data["MinimumPrice"]) < minimumPreviousPrice and data["State"]>=state:
            minimumPreviousPrice = util.getPrice(data["MinimumPrice"])

    return minimumPreviousPrice

def getMaximumPreviousPrice(departureDate, state, datas):
    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    maximumPreviousPrice = util.getPrice(specificDatas[0]["MinimumPrice"])
    for data in specificDatas:
        if util.getPrice(data["MinimumPrice"]) > maximumPreviousPrice and data["State"]>=state:
            maximumPreviousPrice = util.getPrice(data["MinimumPrice"])

    return maximumPreviousPrice


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 12),  # 96x96 input pixels per batch
    hidden_num_units=7,  # number of units in hidden layer
    output_nonlinearity=lasagne.nonlinearities.sigmoid,  # output layer uses identity function
    output_num_units=1,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.005,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=25,  # we want to train this many epochs
    verbose=0,
    )


def mainNNLearning():
    X_train = np.load('inputNN/X_train.npy')
    y_train = np.load('inputNN/y_train.npy')
    # deal with unbalanced data
    X_train, y_train = dealingUnbalancedData(X_train, y_train)

    # train the NN model
    net1.fit(X_train, y_train)

    X_test = np.load('inputNN/X_test.npy')
    y_test = np.load('inputNN/y_test.npy')

    # predict the test data
    y_pred = net1.predict(X_test)

    # 1 for buy, 0 for wait
    median = np.median(y_pred)
    mean = np.mean(y_pred)
    y_pred[y_pred>=median] = 1
    y_pred[y_pred<median] = 0

    print "Number of buy: {}".format(np.count_nonzero(y_pred))
    print "Number of wait: {}".format(np.count_nonzero(1-y_pred))
    #print np.concatenate((y_test, y_pred), axis=1)
    #print y_pred.T.tolist()[0]
    #print map(round, y_pred.T.tolist()[0])
    #print len(y_pred.T.tolist())
    return X_test, y_pred


def dealingUnbalancedData(X_train, y_train):
    """
    Dealing with unbalanced training data
    """
    len0 = np.count_nonzero(1-y_train)
    len1 = np.count_nonzero(y_train)
    dup = int(len0/len1)

    y1 = y_train[np.where(y_train==1)[0], :]
    X1 = X_train[np.where(y_train==1)[0], :]

    X1 = np.tile(X1, (dup-1,1))
    y1 = np.tile(y1, (dup-1,1))

    X_train = np.concatenate((X_train, X1), axis=0)
    y_train = np.concatenate((y_train, y1), axis=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)  # shuffle train data

    return X_train, y_train

def evaluateOneRoute(filePrefix="BCN_BUD"):
    """
    Evaluate one route for one time
    :param filePrefix: route
    :return: average price
    """
    X_test, y_pred = mainNNLearning()
    y_test_price = np.load('inputNN/y_test_price.npy')
    y_price = np.empty(shape=(0, 1))
    for i in range(y_test_price.shape[0]):
        price = [util.getPrice(y_test_price[i, 0])]
        y_price = np.concatenate((y_price, [price]), axis=0)

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix = np.concatenate((X_test, y_pred, y_price), axis=1)

    flightNum = routes.index(filePrefix)

    evalMatrix = evalMatrix[np.where(evalMatrix[:, flightNum]==1)[0], :]

    # group by the feature 8: departure date
    departureDates = np.unique(evalMatrix[:, 8])

    departureLen = len(departureDates)
    latestBuyDate = 7 # define the latest buy date state
    totalPrice = 0
    totalMinimumPrice = 0 # get the best performance
    totalMaximumPrice = 0 # get the worst performance
    for departureDate in departureDates:
        state = latestBuyDate
        for i in range(evalMatrix.shape[0]):
            # if no entry is buy, then buy the latest one
            if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] == latestBuyDate:
                latestPrice = evalMatrix[i, 13]
            # if many entries is buy, then buy the first one
            if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] >= state and evalMatrix[i, 12] == 1:
                state = evalMatrix[i, 9]
                price = evalMatrix[i, 13]

        try:
            totalPrice += price
        except:
            print "Price is not find, buy the latest one"
            totalPrice += latestPrice

    avgPrice = totalPrice * 1.0 / departureLen
    print "One Time avg price: {}".format(avgPrice)
    return avgPrice

def getBestAndWorstPrice(filePrefix):
    """
    If you want to get the maximum and minimum price from the stored json file, use this function
    :param filePrefix: route prefix
    :return: maximum and minimum price dictionary
    """
    with open('results/data_NNlearing_minimumPrice_{:}.json'.format(filePrefix), 'r') as infile:
        minimumPrice = json.load(infile)
    with open('results/data_NNlearing_maximumPrice_{:}.json'.format(filePrefix), 'r') as infile:
        maximumPrice = json.load(infile)

    return minimumPrice, maximumPrice

def evaluateOneRouteForMultipleTimes(filePrefix="BCN_BUD"):
    """
    Rune the evaluation multiple times(here 100), to get the avarage performance
    :param filePrefix: route
    :return: average price
    """
    minimumPrice, maximumPrice = getBestAndWorstPrice(filePrefix)
    minimumPrice = sum(minimumPrice.values()) * 1.0 / len(minimumPrice)
    maximumPrice = sum(maximumPrice.values()) * 1.0 / len(maximumPrice)
    print minimumPrice
    print maximumPrice

    totalPrice = 0
    for i in range(100):
        price = evaluateOneRoute(filePrefix)
        totalPrice += price

    avgPrice = totalPrice * 1.0 / 100

    print "100 times avg price: {}".format(avgPrice)
    print "Minimum price: {}".format(minimumPrice)
    print "Maximum price: {}".format(maximumPrice)
    return avgPrice

if __name__ == "__main__":
    evaluateOneRouteForMultipleTimes(routes[0])





