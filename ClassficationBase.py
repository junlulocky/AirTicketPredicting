# system library
import numpy as np
import json

# user-library
import load_data
import util

# third-party library
from sklearn.utils import shuffle
from sklearn import preprocessing



class ClassificationBase(object):

    def __init__(self, isTrain):
        # indicate it is train data or not
        self.isTrain = isTrain
        # route prefix
        self.routes = ["BCN_BUD",  # route 1
                      "BUD_BCN",  # route 2
                      "CRL_OTP",  # route 3
                      "MLH_SKP",  # route 4
                      "MMX_SKP",  # route 5
                      "OTP_CRL",  # route 6
                      "SKP_MLH",  # route 7
                      "SKP_MMX"]  # route 8
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
        # output: prediction(buy or wait); output_price: price
        # load training datasets
        self.X_train = np.load('inputClf/X_train.npy')
        self.y_train = np.load('inputClf/y_train.npy')
        self.y_train_price = np.load('inputClf/y_train_price.npy')

        # deal with unbalanced data
        #self.X_train, self.y_train = self.dealingUnbalancedData(self.X_train, self.y_train)

        # load test datasets
        if isTrain:
            self.X_test = np.load('inputClf/X_train.npy')
            self.y_test = np.load('inputClf/y_train.npy')
            self.y_test_price = np.load('inputClf/y_train_price.npy')
            self.y_pred = np.empty(shape=(self.y_test.shape[0],1))

            # choose the dates whose departureDate-queryDate gaps is larger than 20
            self.y_test = self.y_test[np.where(self.X_test[:, 8]>20)[0], :]
            self.y_test_price = self.y_test_price[np.where(self.X_test[:, 8]>20)[0], :]
            self.y_pred = self.y_pred[np.where(self.X_test[:, 8]>20)[0], :]
            self.X_test = self.X_test[np.where(self.X_test[:, 8]>20)[0], :]
        else:
            self.X_test = np.load('inputClf/X_test.npy')
            self.y_test = np.load('inputClf/y_test.npy')
            self.y_test_price = np.load('inputClf/y_test_price.npy')
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

        #self.X_train = np.concatenate((self.X_train, self.y_train_price), axis=1)
        #self.X_test = np.concatenate((self.X_test, self.y_test_price), axis=1)


        np.save('inputClf/X_train', self.X_train)
        np.save('inputClf/y_train', self.y_train)
        np.save('inputClf/y_train_price', self.y_train_price)
        np.save('inputClf/X_test', self.X_test)
        np.save('inputClf/y_test', self.y_test)
        np.save('inputClf/y_test_price', self.y_test_price)

    def Standardization(self):
        scaled = preprocessing.scale(self.X_train[:, 10:12])
        self.X_train[:, 10:12] = scaled

        scaled = preprocessing.scale(self.X_test[:, 10:12])
        self.X_test[:, 10:12] = scaled

    def load(self, dataset="large data set"):
        """
        Load the data for classification
        :param dataset: dataset
        :return: X_train, y_train, X_test, y_test
        """
        isOneOptimalState = False
        # Construct the input data
        d = 12
        X_train = np.empty(shape=(0, d))
        y_train = np.empty(shape=(0,1))
        y_train_price = np.empty(shape=(0,1))
        X_test = np.empty(shape=(0,d))
        y_test = np.empty(shape=(0,1))
        y_test_price = np.empty(shape=(0,1))

        for filePrefix in self.routes:
            datas = load_data.load_data_with_prefix_and_dataset(filePrefix, dataset)
            for data in datas:
                print "Construct route {}, State {}, departureDate {}...".format(filePrefix, data["State"], data["Date"])
                x_i = []
                # feature 1: flight number -> dummy variables
                for i in range(len(self.routes)):
                    """
                    !!!need to change!
                    """
                    if i == self.routes.index(filePrefix):
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
                minimumPreviousPrice = self.getMinimumPreviousPrice(data["Date"], state, datas)
                x_i.append(minimumPreviousPrice)

                # feature 5: maximum price before the observed date
                maximumPreviousPrice = self.getMaximumPreviousPrice(data["Date"], state, datas)
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

    def dealingUnbalancedData(self):
        """
        Dealing with unbalanced training data
        """
        len0 = np.count_nonzero(1-self.y_train)
        len1 = np.count_nonzero(self.y_train)
        dup = int(len0/len1)
        dup = int(dup * 1.5)  # change this value

        X1 = self.X_train[np.where(self.y_train==1)[0], :]
        y1 = self.y_train[np.where(self.y_train==1)[0], :]
        y2 = self.y_train_price[np.where(self.y_train==1)[0], :]

        X1 = np.tile(X1, (dup-1,1))
        y1 = np.tile(y1, (dup-1,1))
        y2 = np.tile(y2, (dup-1,1))

        self.X_train = np.concatenate((self.X_train, X1), axis=0)
        self.y_train = np.concatenate((self.y_train, y1), axis=0)
        self.y_train_price = np.concatenate((self.y_train_price, y2), axis=0)
        # shuffle train data
        self.X_train, self.y_train, self.y_train_price = shuffle(self.X_train, self.y_train, self.y_train_price, random_state=42)


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
        # feature 4: output(buy or wait); feature 5: prediction
        # feature 7: current price
        X_test = X_test[:, 8:15]

        # group by the feature: departure date
        departureDates_test = np.unique(X_test[:, 0])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        length_test = []
        for departureDate in departureDates_test:
            indexs = np.where(X_test[:, 0]==departureDate)[0]
            datas = X_test[indexs, :]
            length_test.append(len(datas))
            print departureDate
            print datas

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
        # feature 4: prediction(buy or wait).
        X_train = X_train[:, 8:14]

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

    def evaluateOneRoute(self, filePrefix="BCN_BUD"):
        """
        Evaluate one route for one time
        :param filePrefix: route
        :return: average price
        """
        self.training()
        self.predict()
        #X_test, y_pred = self.predict()
        X_test = self.X_test
        y_pred = self.y_pred
        y_pred = y_pred.reshape((y_pred.shape[0], 1))
        y_test_price = self.y_test_price #np.load('inputClf/y_test_price.npy')
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
        with open('random_train/randomPrice_{:}.json'.format(filePrefix), 'r') as infile:
            randomPrice = json.load(infile)

        return minimumPrice, maximumPrice, randomPrice

    def evaluateOneRouteForMultipleTimes(self, filePrefix="BCN_BUD"):
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

        if self.isTrain:
            print "20 times avg price: {}".format(avgPrice)
            print "Minimum price: {}".format(self.minPrices_train[flightNum])
            print "Random price: {}".format(self.randomPrices_train[flightNum])
            performance = (self.randomPrices_train[flightNum] - avgPrice) / self.randomPrices_train[flightNum] * 100
            print "Performance: {}".format(performance)
            maxPerformance = (self.randomPrices_train[flightNum] - self.minPrices_train[flightNum]) / self.randomPrices_train[flightNum] * 100
            print "Max Performance: {}".format(maxPerformance)
            normalizedPefor = performance / maxPerformance * 100
            print "Normalized Performance: {}".format(normalizedPefor)
        else:
            print "20 times avg price: {}".format(avgPrice)
            print "Minimum price: {}".format(self.minPrices_test[flightNum])
            print "Random price: {}".format(self.randomPrices_test[flightNum])
            performance = (self.randomPrices_test[flightNum] - avgPrice) / self.randomPrices_test[flightNum] * 100
            print "Performance: {}".format(performance)
            maxPerformance = (self.randomPrices_test[flightNum] - self.minPrices_test[flightNum]) / self.randomPrices_test[flightNum] * 100
            print "Max Performance: {}".format(maxPerformance)
            normalizedPefor = performance / maxPerformance * 100
            print "Normalized Performance: {}".format(normalizedPefor)
        #print "Minimum price: {}".format(minimumPrice)
        #print "Maximum price: {}".format(maximumPrice)
        #print "Random price: {}".format(randomPrice)
        return avgPrice

