# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet



class ClassificationNN(ClassficationBase.ClassificationBase):
    def __init__(self):
        super(ClassificationNN, self).__init__()
        # deal with unbalanced data
        self.X_train, self.y_train = self.dealingUnbalancedData(self.X_train, self.y_train)

        self.net1 = NeuralNet(
                        layers=[  # three layers: one hidden layer
                            ('input', layers.InputLayer),
                            ('hidden', layers.DenseLayer),
                            #('hidden2', layers.DenseLayer),
                            ('output', layers.DenseLayer),
                            ],
                        # layer parameters:
                        input_shape=(None, 12),  # 96x96 input pixels per batch
                        hidden_num_units=7,  # number of units in hidden layer
                        #hidden2_num_units=3,  # number of units in hidden layer
                        output_nonlinearity=lasagne.nonlinearities.sigmoid,  # output layer uses sigmoid function
                        output_num_units=1,  # 30 target values

                        # optimization method:
                        update=nesterov_momentum,
                        update_learning_rate=0.005,
                        update_momentum=0.9,

                        regression=True,  # flag to indicate we're dealing with regression problem
                        max_epochs=25,  # we want to train this many epochs
                        verbose=0,
                        )


    def training(self):
        # train the NN model
        self.net1.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        y_pred_train = self.net1.predict(self.X_train)
        self.y_pred = self.net1.predict(self.X_test)

        # 1 for buy, 0 for wait
        median = np.median(y_pred_train)
        mean = np.mean(y_pred_train)
        self.y_pred[self.y_pred>=median] = 1  # change this threshold
        self.y_pred[self.y_pred<median] = 0

        print "Number of buy: {}".format(np.count_nonzero(self.y_pred))
        print "Number of wait: {}".format(np.count_nonzero(1-self.y_pred))
        #print np.concatenate((y_test, y_pred), axis=1)
        #print y_pred.T.tolist()[0]
        #print map(round, y_pred.T.tolist()[0])
        #print len(y_pred.T.tolist())
        return self.X_test, self.y_pred

    def evaluateOneRoute(self, filePrefix="BCN_BUD"):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """
        self.training()
        X_test, y_pred = self.predict()
        y_test_price = np.load('inputClf/y_test_price.npy')
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
        flightNum = self.routes.index(filePrefix)

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

    def getBestAndWorstPrice(self, filePrefix):
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

    def evaluateOneRouteForMultipleTimes(self, filePrefix="BCN_BUD"):
        """
        Rune the evaluation multiple times(here 100), to get the avarage performance
        :param filePrefix: route
        :return: average price
        """
        minimumPrice, maximumPrice = self.getBestAndWorstPrice(filePrefix)
        minimumPrice = sum(minimumPrice.values()) * 1.0 / len(minimumPrice)
        maximumPrice = sum(maximumPrice.values()) * 1.0 / len(maximumPrice)

        totalPrice = 0
        for i in range(20):
            np.random.seed(i*i) # do not forget to set seed for the weight initialization
            price = self.evaluateOneRoute(filePrefix)
            totalPrice += price

        avgPrice = totalPrice * 1.0 / 20

        print "20 times avg price: {}".format(avgPrice)
        print "Minimum price: {}".format(minimumPrice)
        print "Maximum price: {}".format(maximumPrice)
        return avgPrice