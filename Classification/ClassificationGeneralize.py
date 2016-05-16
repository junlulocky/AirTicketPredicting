# system library
import numpy as np
import json

# user-library
import ClassficationBase
import util


# third-party library
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class ClassificationGeneralize(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain):
        super(ClassificationGeneralize, self).__init__(isTrain)
        # data preprocessing
        self.dataPreprocessing()

        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")

        # load the general data
        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price

        # feature 12: output; feature 13: current price
        # feature 14: flight index
        self.X_general = np.load('inputGeneralClfParsed/X_train.npy')
        self.y_general = np.load('inputGeneralClfParsed/y_train.npy')
        self.y_general = self.y_general.reshape((self.y_general.shape[0], 1))
        self.y_general_price = np.load('inputGeneralClfParsed/y_train_price.npy')
        self.y_general_price = self.y_general_price.reshape((self.y_general_price.shape[0], 1))
        self.y_general_index = np.load('inputGeneralClfParsed/y_index.npy')
        self.y_general_index = self.y_general_index.reshape((self.y_general_index.shape[0], 1))



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

        # train and predict on the fly
        self.training()
        self.predict()


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
        self.y_pred = self.ada.predict(self.X_general)
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        self.y_general = self.y_general.reshape((self.y_general.shape[0], 1))

        err = 1 - np.sum(self.y_general == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error Rate: {}".format(err)
        return np.concatenate((self.y_general, self.y_pred), axis=1)

    def pickMinTicketByNumpy(self, flightNum):


    def getMinTicketPriceForAllRoutesByNumpy(self):
        self.minPrices_general = []
        for route in range(len(self.routes_general)):
            # take one route
            # load the general data
            # feature 0~7: flight number dummy variables
            # feature 8: departure date; feature 9: observed date state;
            # feature 10: minimum price; feature 11: maximum price
            # feature 12: current price
            X_general = self.X_general[np.where(self.y_general_index==flightNum)[0], :]
            y_general_price = self.y_general_price[np.where(self.y_general_index==flightNum)[0], :]

            evalMatrix = np.concatenate((X_general, y_general_price), axis=1)

            # take the departure date 20 days after the first observed date
            evalMatrix = evalMatrix[np.where(evalMatrix[:, 8]>20)[0], :]


            totalPrice = 0;
            len = 0;
            departureDates = np.unique(evalMatrix[:, 8])
            for departureDate in departureDates:
                tmpMatrix = evalMatrix[np.where(evalMatrix[:, 8]==departureDate)[0], :]
                tmpMatrix = tmpMatrix[:, 12]
                tmpMatrix = tmpMatrix.reshape((tmpMatrix.shape[0], 1))
                totalPrice += tmpMatrix.min()

            avgPrice = totalPrice * 1.0 / departureDates.shape[0]
            return avgPrice
            self.minPrices_general.append(avgPrice)
            print avgPrice

    def pickRandomTicketByNumpy(self, flightNum):
        # take one route
        # load the general data
        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        # feature 12: current price
        X_general = self.X_general[np.where(self.y_general_index==flightNum)[0], :]
        y_general_price = self.y_general_price[np.where(self.y_general_index==flightNum)[0], :]

        evalMatrix = np.concatenate((X_general, y_general_price), axis=1)

        # take the departure date 20 days after the first observed date
        evalMatrix = evalMatrix[np.where(evalMatrix[:, 8]>20)[0], :]

        totalPrice = 0;
        len = 0;
        departureDates = np.unique(evalMatrix[:, 8])
        for departureDate in departureDates:
            tmpMatrix = evalMatrix[np.where(evalMatrix[:, 8]==departureDate)[0], :]
            tmpMatrix = tmpMatrix[:, 12]
            if tmpMatrix.shape[0] > 30:
                np.random.shuffle(tmpMatrix)
                tmpMatrix = tmpMatrix.reshape((tmpMatrix.shape[0], 1))
                totalPrice += np.sum(tmpMatrix[0:30, :])
                len += 30
            else:
                totalPrice += np.sum(tmpMatrix)
                len += tmpMatrix.shape[0]

        avgPrice = totalPrice * 1.0 / len
        return avgPrice




    def getRandomTicketPriceForAllRoutesByNumpy(self):
        self.randomPrices_general = []
        for route in range(len(self.routes_general)):
            avgPrice = self.pickRandomTicketByNumpy(route)
            self.randomPrices_general.append(avgPrice)
            print avgPrice

    def evaluateGeneralOneRoute(self, filePrefix):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """

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

        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        # fearure 12: prediction(buy or wait); feature 13: price
        evalMatrix = np.concatenate((X_general, y_pred, y_general_price), axis=1)


        # group by the feature 8: departure date
        departureDates = np.unique(evalMatrix[:, 8])

        departureLen = len(departureDates)
        latestBuyDate = 2 # define the latest buy date state
        totalPrice = 0
        for departureDate in departureDates:
            state = latestBuyDate # update the state for every departure date evaluation
            global isFound # indicate whether some entries is predicted to be buy
            isFound = 0
            for i in range(evalMatrix.shape[0]):
                # if no entry is buy, then buy the latest one
                if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] == latestBuyDate:
                    latestPrice = evalMatrix[i, 13]
                # if many entries is buy, then buy the first one
                if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] >= state and evalMatrix[i, 12] == 1:
                    isFound = 1
                    state = evalMatrix[i, 9]
                    price = evalMatrix[i, 13]

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
        minPrices = [35.607128463475924, 66.996571428571357, 47.116582278481076, 38.72333333333318, 57.243768844221435, 43.929999999999836, 41.639168765743072, 64.97784810126582, 97.086272040302262, 52.046666666666603, 60.514999999999858, 34.822000000000102]
        randomPrices = [43.913993136166347, 77.267677018633634, 55.72995343658269, 46.317650630546659, 67.783146824997289, 55.237927612376197, 49.547365499142018, 73.845303168336869, 115.6405006547799, 63.370157034442705, 74.28428571428573, 41.287631947687849]

        minimumPrice = minPrices[flightNum]
        randomPrice = randomPrices[flightNum]

        timesToRun = 1 # if it is neural network, please change this number to 20 or more
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

