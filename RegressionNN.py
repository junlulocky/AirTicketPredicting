# system library
import numpy as np
import json

# user-library
import RegressionBase


# third-party library
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit



class RegressionNN(RegressionBase.RegressionBase):
    def __init__(self):
        super(RegressionNN, self).__init__()
        # data preprocessing
        #self.dataPreprocessing()

        self.net1 = NeuralNet(
                        layers=[  # three layers: one hidden layer
                            ('input', layers.InputLayer),
                            ('hidden', layers.DenseLayer),
                            #('hidden2', layers.DenseLayer),
                            #('hidden3', layers.DenseLayer),
                            ('output', layers.DenseLayer),
                            ],
                        # layer parameters:
                        input_shape=(None, 13),  # input dimension is 13
                        hidden_num_units=6,  # number of units in hidden layer
                        #hidden2_num_units=8,  # number of units in hidden layer
                        #hidden3_num_units=4,  # number of units in hidden layer
                        output_nonlinearity=None,  # output layer uses sigmoid function
                        output_num_units=1,  # output dimension is 1

                        # obejctive function
                        objective_loss_function = lasagne.objectives.squared_error,

                        # optimization method:
                        update=lasagne.updates.nesterov_momentum,
                        update_learning_rate=0.002,
                        update_momentum=0.4,

                        # use 25% as validation
                        train_split=TrainSplit(eval_size=0.2),

                        regression=True,  # flag to indicate we're dealing with regression problem
                        max_epochs=100,  # we want to train this many epochs
                        verbose=0,
                        )

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the NN model
        self.net1.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.net1.predict(self.X_test)

    def evaluateOneRoute(self, filePrefix):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """
        self.training()
        self.predict()

        X_test = self.X_test
        y_pred = self.y_pred
        y_test_price = self.y_test_price
        y_buy = np.zeros(shape=(y_pred.shape[0], y_pred.shape[1]))
        y_buy[np.where((y_test_price<y_pred)==True)[0], :] = 1  # to indicate whether buy or not

        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        # feature 12: current price;
        # fearure 13: prediction(buy or wait); feature 14: current price
        evalMatrix = np.concatenate((X_test, y_buy, y_test_price), axis=1)

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
                    latestPrice = evalMatrix[i, 14]
                # if many entries is buy, then buy the first one
                if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] >= state and evalMatrix[i, 13] == 1:
                    state = evalMatrix[i, 9]
                    price = evalMatrix[i, 14]

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

    def getBestAndWorstAndRandomPrice(self, filePrefix):
        """
        If you want to get the maximum and minimum price from the stored json file, use this function
        :param filePrefix: route prefix
        :return: maximum and minimum price dictionary
        """
        with open('results/data_NNlearing_minimumPrice_{:}.json'.format(filePrefix), 'r') as infile:
            minimumPrice = json.load(infile)
        with open('results/data_NNlearing_maximumPrice_{:}.json'.format(filePrefix), 'r') as infile:
            maximumPrice = json.load(infile)
        with open('randomPrice/randomPrice_{:}.json'.format(filePrefix), 'r') as infile:
            randomPrice = json.load(infile)

        return minimumPrice, maximumPrice, randomPrice

    def evaluateOneRouteForMultipleTimes(self, filePrefix):
        """
        Rune the evaluation multiple times(here 100), to get the avarage performance
        :param filePrefix: route
        :return: average price
        """
        # route index
        flightNum = self.routes.index(filePrefix)

        # get the maximum, minimum, and randomly picked prices
        minimumPrice, maximumPrice, randomPrice = self.getBestAndWorstAndRandomPrice(filePrefix)
        minimumPrice = sum(minimumPrice.values()) * 1.0 / len(minimumPrice) * self.currency[flightNum]
        maximumPrice = sum(maximumPrice.values()) * 1.0 / len(maximumPrice) * self.currency[flightNum]
        randomPrice = randomPrice * self.currency[flightNum]

        totalPrice = 0
        for i in range(20):
            np.random.seed(i*i) # do not forget to set seed for the weight initialization
            price = self.evaluateOneRoute(filePrefix)
            totalPrice += price

        avgPrice = totalPrice * 1.0 / 20

        print "20 times avg price: {}".format(avgPrice)
        print "Minimum price: {}".format(minimumPrice)
        print "Maximum price: {}".format(maximumPrice)
        print "Random price: {}".format(randomPrice)
        return avgPrice