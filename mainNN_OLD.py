# system library
import numpy as np
import json

# user-library
import load_data
import util

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
    isOneOptimalState = False
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

            if isOneOptimalState:
                # Method 1: only 1 entry is buy
                optimalState = load_data.getOptimalState(specificDatas)
                if data["State"] == optimalState:
                   y_i = [1]
            else:
                # Method 2: multiple entries can be buy
                minPrice = load_data.getMinimumPrice(specificDatas)
                if util.getPrice(data["MinimumPrice"]) == minPrice:
                    y_i = [1]


            # keep price info
            y_price = [util.getPrice(data["MinimumPrice"])]

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

    if isOneOptimalState:
        np.save('inputNN_1Buy/X_train', X_train)
        np.save('inputNN_1Buy/y_train', y_train)
        np.save('inputNN_1Buy/y_train_price', y_train_price)
        np.save('inputNN_1Buy/X_test', X_test)
        np.save('inputNN_1Buy/y_test', y_test)
        np.save('inputNN_1Buy/y_test_price', y_test_price)
    else:
        np.save('inputNN_NBuy/X_train', X_train)
        np.save('inputNN_NBuy/y_train', y_train)
        np.save('inputNN_NBuy/y_train_price', y_train_price)
        np.save('inputNN_NBuy/X_test', X_test)
        np.save('inputNN_NBuy/y_test', y_test)
        np.save('inputNN_NBuy/y_test_price', y_test_price)


    return X_train, y_train, X_test, y_test



def getMinimumPreviousPrice(departureDate, state, datas):
    """
    Get the minimum previous price, corresponding to the departure date and the observed date
    :param departureDate: departure date
    :param state: observed date
    :param datas: datasets
    :return: minimum previous price
    """
    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    minimumPreviousPrice = util.getPrice(specificDatas[0]["MinimumPrice"])
    for data in specificDatas:
        if util.getPrice(data["MinimumPrice"]) < minimumPreviousPrice and data["State"]>=state:
            minimumPreviousPrice = util.getPrice(data["MinimumPrice"])

    return minimumPreviousPrice

def getMaximumPreviousPrice(departureDate, state, datas):
    """
    Get the maximum previous price, corresponding to the departure date and the observed date
    :param departureDate: departure date
    :param state: observed date
    :param datas: datasets
    :return: maximum previous price
    """
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
        #('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 12),  # input dimension is 12
    hidden_num_units=7,  # number of units in hidden layer
    #hidden2_num_units=3,  # number of units in hidden layer
    output_nonlinearity=lasagne.nonlinearities.sigmoid,  # output layer uses sigmoid function
    output_num_units=1,  # output dimension is 1

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.005,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=25,  # we want to train this many epochs
    verbose=0,
    )


def mainNNLearning():
    X_train = np.load('inputNN_NBuy/X_train.npy')
    y_train = np.load('inputNN_NBuy/y_train.npy')
    # deal with unbalanced data
    X_train, y_train = dealingUnbalancedData(X_train, y_train)

    # train the NN model
    net1.fit(X_train, y_train)

    X_test = np.load('inputNN_NBuy/X_test.npy')
    y_test = np.load('inputNN_NBuy/y_test.npy')

    # predict the test data
    y_pred_train = net1.predict(X_train)
    y_pred = net1.predict(X_test)

    # 1 for buy, 0 for wait
    median = np.median(y_pred_train)
    mean = np.mean(y_pred_train)
    mm2 = (median+mean)/2
    mm3 = (1.5*median + 2*mean)/3.5
    y_pred[y_pred>=median] = 1  # change this threshold
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
    dup = int(dup * 1.5)  # change this value

    X1 = X_train[np.where(y_train==1)[0], :]
    y1 = y_train[np.where(y_train==1)[0], :]

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
    y_test_price = np.load('inputNN_NBuy/y_test_price.npy')
    """
    y_price = np.empty(shape=(0, 1))
    for i in range(y_test_price.shape[0]):
        price = [util.getPrice(y_test_price[i, 0])]
        y_price = np.concatenate((y_price, [price]), axis=0)
    """

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix = np.concatenate((X_test, y_pred, y_test_price), axis=1)

    # route index
    flightNum = routes.index(filePrefix)

    evalMatrix = evalMatrix[np.where(evalMatrix[:, flightNum]==1)[0], :]

    # group by the feature 8: departure date
    departureDates = np.unique(evalMatrix[:, 8])

    departureLen = len(departureDates)
    latestBuyDate = 7 # define the latest buy date state
    totalPrice = 0
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
            if state >= latestBuyDate:
                price = 0 + price # do not forget this step, or the totalPrice will add random number
                totalPrice += price
            else:
                totalPrice += latestPrice
        except:
            print "Price is not find, buy the latest one {}".format(latestPrice)
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

    totalPrice = 0
    for i in range(20):
        np.random.seed(i*i) # do not forget to set seed for the weight initialization
        price = evaluateOneRoute(filePrefix)
        totalPrice += price

    avgPrice = totalPrice * 1.0 / 20

    print "20 times avg price: {}".format(avgPrice)
    print "Minimum price: {}".format(minimumPrice)
    print "Maximum price: {}".format(maximumPrice)
    return avgPrice

if __name__ == "__main__":
    # evaluate one route for 20 times to get the average price
    evaluateOneRouteForMultipleTimes(routes[4])










