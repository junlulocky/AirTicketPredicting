# system library
import numpy as np
import json

# user-library
import load_data
import util

# third-party library
from sklearn.utils import shuffle
from sklearn import preprocessing



class RegressionBase(object):

    def __init__(self):
        # route prefix
        self.routes = ["BCN_BUD",  # route 1
                      "BUD_BCN",  # route 2
                      "CRL_OTP",  # route 3
                      "MLH_SKP",  # route 4
                      "MMX_SKP",  # route 5
                      "OTP_CRL",  # route 6
                      "SKP_MLH",  # route 7
                      "SKP_MMX"]  # route 8
        # for currency change
        self.currency = [1,      # route 1 - Euro
                         0.0032, # route 2 - Hungarian Forint
                         1,      # route 3 - Euro
                         1,      # route 4 - Euro
                         0.12,   # route 5 - Swedish Krona
                         0.25,   # route 6 - Romanian Leu
                         0.018,  # route 7 - Macedonian Denar
                         0.018   # route 8 - Macedonian Denar
                         ]

        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        # feature 12: current price
        # output: prediction(buy or wait); output_price: price
        # load training datasets
        self.X_train = np.load('inputReg/X_train.npy')
        self.y_train = np.load('inputReg/y_train.npy')
        self.y_train_price = np.load('inputReg/y_train_price.npy')

        # load test datasets
        self.X_test = np.load('inputReg/X_test.npy')
        self.y_test = np.load('inputReg/y_test.npy')
        self.y_test_price = np.load('inputReg/y_test_price.npy')
        self.y_pred = np.empty(shape=(self.y_test.shape[0],1))

    def priceNormalize(self):
        """
        Different routes have different units for the price, normalize it as Euro.
        :return: NA
        """
        # normalize feature 10, feature 11, feature 13
        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price
        # fearure 12: prediction(buy or wait); feature 13: price
        evalMatrix_train = np.concatenate((self.X_train, self.y_train, self.y_train_price), axis=1)
        evalMatrix_test = np.concatenate((self.X_test, self.y_test, self.y_test_price), axis=1)

        matrixTrain = np.empty(shape=(0, evalMatrix_train.shape[1]))
        matrixTest = np.empty(shape=(0, evalMatrix_train.shape[1]))
        for i in range(len(self.routes)):
            evalMatrix = evalMatrix_train[np.where(evalMatrix_train[:, i]==1)[0], :]
            evalMatrix[:, 10] *= self.currency[i]
            evalMatrix[:, 11] *= self.currency[i]
            evalMatrix[:, 13] *= self.currency[i]
            matrixTrain = np.concatenate((matrixTrain, evalMatrix), axis=0)

            evalMatrix = evalMatrix_test[np.where(evalMatrix_test[:, i]==1)[0], :]
            evalMatrix[:, 10] *= self.currency[i]
            evalMatrix[:, 11] *= self.currency[i]
            evalMatrix[:, 13] *= self.currency[i]
            matrixTest = np.concatenate((matrixTest, evalMatrix), axis=0)

        self.X_train = matrixTrain[:, 0:12]
        self.y_train = matrixTrain[:, 12]
        self.y_train_price = matrixTrain[:, 13]

        self.X_test = matrixTest[:, 0:12]
        self.y_test = matrixTest[:, 12]
        self.y_test_price = matrixTest[:, 13]

        self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))
        self.y_train_price = self.y_train_price.reshape((self.y_train_price.shape[0], 1))
        self.y_test = self.y_test.reshape((self.y_test.shape[0], 1))
        self.y_test_price = self.y_test_price.reshape((self.y_test_price.shape[0], 1))

    def Standardization(self):
        # feature 10: minimum price so far; feature 11: maximum price so far
        # feature 12: current price
        scaled = preprocessing.scale(self.X_train[:, 10:13])
        self.X_train[:, 10:13] = scaled

        scaled = preprocessing.scale(self.X_test[:, 10:13])
        self.X_test[:, 10:13] = scaled

    def getRegressionOutput(self):
        """
        Get the regression output formula from the classification datasets.
        :return: Save the regression datasets into inputReg
        """

        # Construct train data
        X_tmp = np.empty(shape=(0, 14))
        for flightNum in range(len(self.routes)):
            # concatenate the buy or wait info to get the total datas
            y_train = self.y_train.reshape((self.y_train.shape[0],1))
            y_train_price = self.y_train_price.reshape((self.y_train_price.shape[0],1))

            X_train = np.concatenate((self.X_train, y_train, y_train_price), axis=1)

            # choose one route datas
            X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

            # remove dummy variables
            # feature 8: departure date;  feature 9: observed date state
            # feature 10: minimum price; feature 11: maximum price
            # feature 12: prediction(buy or wait); feature 13: current price
            X_train = X_train[:, 0:14]

            # group by the feature: departure date
            departureDates_train = np.unique(X_train[:, 8])

            # get the final datas, the observed data state should be from large to small(i.e. for time series)
            for departureDate in departureDates_train:
                indexs = np.where(X_train[:, 8]==departureDate)[0]
                datas = X_train[indexs, :]
                minPrice = min(datas[:, 10])
                datas[:, 12] = minPrice
                """
                print departureDate
                print minPrice
                print datas
                """
                X_tmp = np.concatenate((X_tmp, datas), axis=0)

        X_train = X_tmp[:, 0:12]
        y_train = X_tmp[:, 12]
        y_train_price = X_tmp[:, 13]
        y_train = y_train.reshape((y_train.shape[0], 1))
        y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))


        X_train = np.concatenate((X_train, y_train_price), axis=1)
        np.save('inputReg/X_train', X_train)
        np.save('inputReg/y_train', y_train)
        np.save('inputReg/y_train_price', y_train_price)


        # Construct test data
        X_tmp = np.empty(shape=(0, 14))
        for flightNum in range(len(self.routes)):
            # concatenate the buy or wait info to get the total datas
            y_test = self.y_test.reshape((self.y_test.shape[0],1))
            y_test_price = self.y_test_price.reshape((self.y_test_price.shape[0],1))

            X_test = np.concatenate((self.X_test, y_test, y_test_price), axis=1)

            # choose one route datas
            X_test = X_test[np.where(X_test[:, flightNum]==1)[0], :]

            # remove dummy variables
            # feature 8: departure date;  feature 9: observed date state
            # feature 10: minimum price; feature 11: maximum price
            # feature 12: prediction(buy or wait); feature 13: current price
            X_test = X_test[:, 0:14]

            # group by the feature: departure date
            departureDates_test = np.unique(X_test[:, 8])

            # get the final datas, the observed data state should be from large to small(i.e. for time series)
            for departureDate in departureDates_test:
                indexs = np.where(X_test[:, 8]==departureDate)[0]
                datas = X_test[indexs, :]
                minPrice = min(datas[:, 10])
                datas[:, 12] = minPrice
                """
                print departureDate
                print minPrice
                print datas
                """
                X_tmp = np.concatenate((X_tmp, datas), axis=0)

        X_test = X_tmp[:, 0:12]
        y_test = X_tmp[:, 12]
        y_test_price = X_tmp[:, 13]
        y_test = y_test.reshape((y_test.shape[0], 1))
        y_test_price = y_test_price.reshape((y_test_price.shape[0], 1))
        X_test = np.concatenate((X_test, y_test_price), axis=1)
        np.save('inputReg/X_test', X_test)
        np.save('inputReg/y_test', y_test)
        np.save('inputReg/y_test_price', y_test_price)

    def getMinimumPreviousPrice(self, departureDate, state, datas):
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

    def getMaximumPreviousPrice(self, departureDate, state, datas):
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

    def visualizePrediction(self, filePrefix):
        """
        Visualize the prediction buy entries for every departure date, for each route
        :param filePrefix: route prefix
        :return: NA
        """
        # route index
        flightNum = self.routes.index(filePrefix)

        # concatenate the buy or wait info to get the total datas
        y_pred = self.y_pred.reshape((self.y_pred.shape[0],1))
        X_test = np.concatenate((self.X_test, self.y_test, y_pred, self.y_test_price), axis=1)

        # choose one route datas
        X_test = X_test[np.where(X_test[:, flightNum]==1)[0], :]

        # remove dummy variables
        # feature 0: departure date;  feature 1: observed date state
        # feature 2: minimum price; feature 3: maximum price
        # feature 4: current price; feature 5: output;
        # feature 6: prediction; feature 7: current price
        X_test = X_test[:, 8:16]

        # group by the feature: departure date
        departureDates_test = np.unique(X_test[:, 0])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        length_test = []
        for departureDate in departureDates_test:
            indexs = np.where(X_test[:, 0]==departureDate)[0]
            datas = X_test[indexs, :]
            length_test.append(len(datas))
            print departureDate
            print datas[:, 3:8]

    def visualizeTrainData(self, filePrefix):
        """
        Visualize the train buy entries for every departure date, for each route
        :param filePrefix: route prefix
        :return: NA
        """
        # route index
        flightNum = self.routes.index(filePrefix)

        # concatenate the buy or wait info to get the total datas
        y_train = self.y_train.reshape((self.y_train.shape[0],1))
        y_train_price = self.y_train_price.reshape((self.y_train_price.shape[0],1))

        X_train = np.concatenate((self.X_train, y_train, y_train_price), axis=1)

        # choose one route datas
        X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

        # remove dummy variables
        # feature 0: departure date;  feature 1: observed date state
        # feature 2: minimum price; feature 3: maximum price
        # feature 4: current price; feature 5: expected minimum price;
        # feature 6: current price
        X_train = X_train[:, 8:15]

        # group by the feature: departure date
        departureDates_train = np.unique(X_train[:, 0])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        length_test = []
        for departureDate in departureDates_train:
            indexs = np.where(X_train[:, 0]==departureDate)[0]
            datas = X_train[indexs, :]
            length_test.append(len(datas))
            print departureDate
            print datas


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
            state = latestBuyDate  # update the state for every departure date evaluation
            global isFound # indicate whether some entries is predicted to be buy
            isFound = 0
            for i in range(evalMatrix.shape[0]):
                # if no entry is buy, then buy the latest one
                if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] == latestBuyDate:
                    latestPrice = evalMatrix[i, 14]
                # if many entries is buy, then buy the first one
                if evalMatrix[i, 8] == departureDate and evalMatrix[i, 9] >= state and evalMatrix[i, 13] == 1:
                    isFound = 1
                    state = evalMatrix[i, 9]
                    price = evalMatrix[i, 14]

            if isFound == 1:
                totalPrice += price
            else:
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


