# system library
import numpy as np
import json

# user-library
from utils import load_data
from utils import util

# third-party library
from sklearn.utils import shuffle
from sklearn import preprocessing



class RegressionBase(object):

    def __init__(self, isTrain, isNN=0):
        # indicate it is train data or not
        self.isTrain = isTrain
        self.isNN = isNN # indicate it is neural network
        # route prefix
        self.routes = ["BCN_BUD",  # route 1
                      "BUD_BCN",  # route 2
                      "CRL_OTP",  # route 3
                      "MLH_SKP",  # route 4
                      "MMX_SKP",  # route 5
                      "OTP_CRL",  # route 6
                      "SKP_MLH",  # route 7
                      "SKP_MMX"]  # route 8

        """
        For the small data set, the datas are from functions in util
        """
        # random price list
        self.randomPrices_train = [68.4391315136,
                             67.4260645161,
                             93.2808545727,
                             77.4751720047,
                             75.0340018399,
                             73.9964736451,
                             105.280932384,
                             97.1720369004]
        self.randomPrices_test = [55.4820634921,
                                  57.8067301587,
                                  23.152037037,
                                  33.3727319588,
                                  35.3032044199,
                                  41.1180555556,
                                  56.3433402062,
                                  60.2546519337]
        # minimum price list
        self.minPrices_train = [44.4344444444,
                               38.9605925926,
                               68.6566666667,
                               49.6566666667,
                               48.2691891892,
                               47.0833333333,
                               68.982,
                               63.1279459459]
        self.minPrices_test = [32.370952381,
                               29.3775238095,
                               11.3788888889,
                               16.5284615385,
                               18.6184615385,
                               14.6111111111,
                               21.5127692308,
                               25.8050769231]

        # maximum price list
        self.maxPrices_train = [115.915925926,
                                126.782814815,
                                144.212222222,
                                129.656666667,
                                141.252972973,
                                149.972222222,
                                174.402,
                                160.91172973
                                ]

        self.maxPrices_test = [126.656666667,
                               168.95847619,
                               93.6011111111,
                               90.5669230769,
                               101.233846154,
                               198.361111111,
                               154.505076923,
                               208.020461538]

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
        self.X_train = np.load('inputReg_small/X_train.npy')
        self.y_train = np.load('inputReg_small/y_train.npy')
        self.y_train_price = np.load('inputReg_small/y_train_price.npy')

        # load test datasets
        if isTrain:
            self.X_test = np.load('inputReg_small/X_train.npy')
            self.y_test = np.load('inputReg_small/y_train.npy')
            self.y_test_price = np.load('inputReg_small/y_train_price.npy')
            self.y_pred = np.empty(shape=(self.y_test.shape[0],1))

            # choose the dates whose departureDate-queryDate gaps is larger than 20
            self.y_test = self.y_test[np.where(self.X_test[:, 8]>20)[0], :]
            self.y_test_price = self.y_test_price[np.where(self.X_test[:, 8]>20)[0], :]
            self.y_pred = self.y_pred[np.where(self.X_test[:, 8]>20)[0], :]
            self.X_test = self.X_test[np.where(self.X_test[:, 8]>20)[0], :]
        else:
            self.X_test = np.load('inputReg_small/X_test.npy')
            self.y_test = np.load('inputReg_small/y_test.npy')
            self.y_test_price = np.load('inputReg_small/y_test_price.npy')

            # """
            # TODO:
            #    Keep only the entries that the observed date is
            #    no longer than 100 days before the departure
            # """
            # self.y_test = self.y_test[np.where(self.X_test[:,9]<=100)[0], :]
            # self.y_test_price = self.y_test_price[np.where(self.X_test[:,9]<=100)[0], :]
            # self.X_test = self.X_test[np.where(self.X_test[:,9]<=100)[0], :]

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
            print "[minimum price, maximum price, current price, output, prediction]"
            print datas[:, 2:7]

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


    def evaluateOneRoute(self, filePrefix, priceTolerance=0):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """

        X_test = self.X_test
        #y_pred = self.y_pred
        y_test_price = self.y_test_price
        y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        y_buy = np.zeros(shape=(y_pred.shape[0], y_pred.shape[1]))
        y_buy[np.where((y_test_price<y_pred+priceTolerance)==True)[0], :] = 1  # to indicate whether buy or not

        # feature 0~7: flight number dummy variables
        # feature 8: departure date; feature 9: observed date state;
        # feature 10: minimum price; feature 11: maximum price;
        # feature 12: current price;
        # fearure 13: prediction(buy or wait); feature 14: current price
        evalMatrix = np.concatenate((X_test, y_buy, y_test_price), axis=1)

        # route index
        flightNum = self.routes.index(filePrefix)

        evalMatrix = evalMatrix[np.where(evalMatrix[:, flightNum]==1)[0], :]

        # group by the feature 8: departure date
        departureDates = np.unique(evalMatrix[:, 8])

        departureLen = len(departureDates)
        latestBuyDate = 11 # define the latest buy date state
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
                #print "departure date: {}, price: {}".format(departureDate, price)
            else:
                totalPrice += latestPrice
                #print "departure date: {}, lastprice: {}".format(departureDate, latestPrice)

        avgPrice = totalPrice * 1.0 / departureLen
        print "One Time avg price: {}".format(avgPrice)
        return avgPrice



    def evaluateOneRouteForMultipleTimes(self, filePrefix, priceTolerance=0, timesToRun=1):
        """
        Rune the evaluation for the given route and run it multiple times(e.g. 100), to get the avarage performance
        :param filePrefix: route prefix
        :param timesToRun: the times to run the evaluation, and get the average.
        :return: average price
        """
        # perform fit and predict
        self.training()
        self.predict()


        # route index
        flightNum = self.routes.index(filePrefix)


        timesToRun = 1 # if it is neural network, please change this number to 20 or more
        if self.isNN:
            timesToRun = 20
        totalPrice = 0
        for i in range(timesToRun):
            np.random.seed(i*i*i) # do not forget to set seed for the weight initialization
            price = self.evaluateOneRoute(filePrefix, priceTolerance)
            totalPrice += price

        avgPrice = totalPrice * 1.0 / timesToRun


        """
        print "20 times avg price: {}".format(avgPrice)
        print "Minimum price: {}".format(minimumPrice)
        print "Maximum price: {}".format(maximumPrice)
        print "Random price: {}".format(randomPrice)
        return avgPrice
        """
        if self.isTrain:
            #print "20 times avg price: {}".format(avgPrice)
            print "TRAIN:"
            print "minimumPrice: {}".format(self.minPrices_train[flightNum])
            print "maximumPrice: {}".format(self.maxPrices_train[flightNum])
            print "randomPrice: {}".format(self.randomPrices_train[flightNum])
            print "avgPredPrice: {}".format(avgPrice)

            performance = (self.randomPrices_train[flightNum] - avgPrice) / self.randomPrices_train[flightNum] * 100
            print "Performance: {}%".format(round(performance,2))
            maxPerformance = (self.randomPrices_train[flightNum] - self.minPrices_train[flightNum]) / self.randomPrices_train[flightNum] * 100
            print "Max Perfor: {}%".format(round(maxPerformance,2))
            normalizedPefor = performance / maxPerformance * 100
            print "Normalized perfor: {}%".format(round(normalizedPefor,2))
        else:
            #print "20 times avg price: {}".format(avgPrice)
            print "TEST:"
            print "minimumPrice: {}".format(self.minPrices_test[flightNum])
            print "maximumPrice: {}".format(self.maxPrices_test[flightNum])
            print "randomPrice: {}".format(self.randomPrices_test[flightNum])
            print "avgPredPrice: {}".format(avgPrice)

            performance = (self.randomPrices_test[flightNum] - avgPrice) / self.randomPrices_test[flightNum] * 100
            print "Performance: {}%".format(round(performance,2))
            maxPerformance = (self.randomPrices_test[flightNum] - self.minPrices_test[flightNum]) / self.randomPrices_test[flightNum] * 100
            print "Max Perfor: {}%".format(round(maxPerformance,2))
            normalizedPefor = performance / maxPerformance * 100
            print "Normalized perfor: {}%".format(round(normalizedPefor,2))

        return (performance, normalizedPefor)

    def evaluateAllRroutes(self):
        """
        Evaluate all the routes, print the performance for every route
        and the average performance for all the routes.
        """
        isTrain = 1 # 1 for train, 0 for test

        performance = 0
        normalizedPerformance = 0
        priceTolerance = 5 # price to be tolerated

        normPerforms = []
        for i in range(8):
            print "Route: {}".format(i)
            [perfor, normaPerfor] = self.evaluateOneRouteForMultipleTimes(self.routes[i], priceTolerance)
            normPerforms.append(normaPerfor)
            performance += perfor
            normalizedPerformance += normaPerfor

        performance = round(performance/8, 2)
        normalizedPerformance = round(normalizedPerformance/8, 2)

        if self.isTrain:
            print "\nTRAIN:"
        else:
            print "\nTEST:"
        print "Average Performance: {}%".format(performance)
        print "Average Normalized Performance: {}%".format(normalizedPerformance)
        print "Normalized Performance Variance: {}".format(np.var(normPerforms))

