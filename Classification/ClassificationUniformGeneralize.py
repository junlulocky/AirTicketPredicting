# system library
import numpy as np
import json

# user-library
import ClassficationBase
from utils import util


# third-party library
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class ClassificationUniformGeneralize(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain):
        super(ClassificationUniformGeneralize, self).__init__(isTrain)
        # data preprocessing
        self.dataPreprocessing()

        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")

        # load the general data
        # feature 0~11: flight number dummy variables
        # feature 12: departure date; feature 13: observed date state;
        # feature 14: minimum price; feature 15: maximum price

        # feature 12: output; feature 13: current price
        # feature 14: flight index
        self.X_general = np.load('inputGeneralClf_small/X_train.npy')
        self.y_general = np.load('inputGeneralClf_small/y_train.npy')
        self.y_general = self.y_general.reshape((self.y_general.shape[0], 1))
        self.y_general_price = np.load('inputGeneralClf_small/y_train_price.npy')
        self.y_general_price = self.y_general_price.reshape((self.y_general_price.shape[0], 1))
        print self.X_general.shape

        self.y_general_index = np.zeros(self.y_general.shape)
        for i in range(self.X_general.shape[0]):
            for idx in range(12):
                if self.X_general[i,idx]==1:
                    self.y_general_index[i,0] = idx


        # define the general routes
        self.routes_general = ["BGY_OTP", # route 1
                                "BUD_VKO", # route 2
                                "CRL_OTP", # route 3
                                "CRL_WAW", # route 4
                                "LTN_OTP", # route 5
                                "LTN_PRG", # route 6
                                "OTP_BGY", # route 7
                                "OTP_CRL", # route 8
                                "OTP_LTN", # route 9
                                "PRG_LTN", # route 10
                                "VKO_BUD", # route 11
                                "WAW_CRL"] # route 12


        """
        define the 8 patterns
        """
        patterns = [[1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,1]]

        """
        get all the 8 inputs
        """
        self.X_generals = []
        for i in range(8):
            tmp = self.X_general[:, 12:16]
            pattern = patterns[i]
            pattern = np.tile(pattern, (tmp.shape[0],1))
            tmp = np.concatenate((pattern, tmp), axis=1)
            self.X_generals.append(tmp)




    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()

    def training(self):
        # train the Decision Tree model
        self.ada.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        y_preds = []

        self.y_pred = np.zeros(self.ada.predict(self.X_generals[0]).shape)

        for i in range(8):
            y_pred = self.ada.predict(self.X_generals[i])
            self.y_pred += y_pred
        self.y_pred = self.y_pred / 8.0
        self.y_pred[self.y_pred >= 0.5] = 1
        self.y_pred[self.y_pred < 0.5] = 0

        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        self.y_general = self.y_general.reshape((self.y_general.shape[0], 1))

        err = 1 - np.sum(self.y_general == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error Rate: {}".format(err)
        return np.concatenate((self.y_general, self.y_pred), axis=1)


    def evaluateGeneralOneRoute(self, filePrefix):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """
        # train and predict on the fly
        self.training()
        self.predict()

        # route index
        flightNum = self.routes_general.index(filePrefix)

        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        X_general = self.X_general
        y_pred = self.y_pred
        y_general_price = self.y_general_price
        y_general_index = self.y_general_index

        # get the data for the one route
        X_general = X_general[np.where(y_general_index==flightNum)[0], :]
        y_pred = y_pred[np.where(y_general_index==flightNum)[0], :]
        y_general_price = y_general_price[np.where(y_general_index==flightNum)[0], :]

        # feature 0~11: flight number dummy variables
        # feature 12: departure date; feature 13: observed date state;
        # feature 14: minimum price; feature 15: maximum price
        # fearure 16: prediction(buy or wait); feature 17: price
        evalMatrix = np.concatenate((X_general, y_pred, y_general_price), axis=1)


        # group by the feature 8: departure date
        departureDates = np.unique(evalMatrix[:, 12])

        departureLen = len(departureDates)
        latestBuyDate = 2 # define the latest buy date state
        totalPrice = 0
        for departureDate in departureDates:
            state = latestBuyDate # update the state for every departure date evaluation
            global isFound # indicate whether some entries is predicted to be buy
            isFound = 0
            for i in range(evalMatrix.shape[0]):
                # if no entry is buy, then buy the latest one
                if evalMatrix[i, 12] == departureDate and evalMatrix[i, 13] == latestBuyDate:
                    latestPrice = evalMatrix[i, 17]
                # if many entries is buy, then buy the first one
                if evalMatrix[i, 12] == departureDate and evalMatrix[i, 13] >= state and evalMatrix[i, 16] == 1:
                    isFound = 1
                    state = evalMatrix[i, 13]
                    price = evalMatrix[i, 17]

            if isFound == 1:
                totalPrice += price
            else:
                totalPrice += latestPrice
        #print isFound
        avgPrice = totalPrice * 1.0 / departureLen
        print "One Time avg price: {}".format(avgPrice)
        return avgPrice

    def evaluateGeneral(self, filePrefix):
        """
        Rune the evaluation multiple times(here 100), to get the avarage performance
        :param filePrefix: route
        :return: average price
        """

        # route index
        flightNum = self.routes_general.index(filePrefix)

        # get the minimum, and randomly picked prices
        minPrices = [12.490000000000006, 46.368000000000009, 17.351111111111116, 23.601111111111116, 26.128888888888895, 29.65666666666667, 17.527777777777779, 29.680555555555557, 71.902777777777771, 32.337999999999994, 34.990000000000002, 22.89777777777779]
        randomPrices = [24.542071668533051, 59.881070866141727, 33.500638297872356, 47.674268426842659, 45.421868131868116, 58.903186813186799, 34.173292273236285, 69.906774916013433, 135.9368131868132, 57.227615384615383, 56.451417322834651, 45.018514851485136]
        maxPrices = [66.378888888888866, 92.521846153846141, 73.045555555555509, 101.93444444444435, 87.101111111111038, 113.76777777777768, 90.583333333333329, 176.20833333333334, 292.04166666666669, 101.71300000000006, 98.528461538461499, 101.92833333333331]




        minimumPrice = minPrices[flightNum]
        randomPrice = randomPrices[flightNum]

        timesToRun = 20 # if it is neural network, please change this number to 20 or more
        totalPrice = 0
        for i in range(timesToRun):
            np.random.seed(i*i) # do not forget to set seed for the weight initialization
            price = self.evaluateGeneralOneRoute(filePrefix)
            totalPrice += price

        avgPrice = totalPrice * 1.0 / timesToRun


        #print "20 times avg price: {}".format(avgPrice)
        print "GENERAL:"
        print "minimumPrice: {}".format(minimumPrice)
        print "randomPrice: {}".format(randomPrice)
        print "avgPredPrice: {}".format(avgPrice)

        performance = (randomPrice - avgPrice) / randomPrice * 100
        print "Performance: {}%".format(round(performance,2))
        maxPerformance = (randomPrice - minimumPrice) / randomPrice * 100
        print "Max Perfor: {}%".format(round(maxPerformance,2))
        normalizedPefor = performance / maxPerformance * 100
        print "Normalized perfor: {}%".format(round(normalizedPefor,2))

        return (performance, normalizedPefor)

