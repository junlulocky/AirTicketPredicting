# system library
import numpy as np

# user-library
import load_data
import util

# third-party library
from sklearn.utils import shuffle



class ClassificationBase(object):

    def __init__(self):
        self.routes = ["BCN_BUD",  # route 1
                      "BUD_BCN",  # route 2
                      "CRL_OTP",  # route 3
                      "MLH_SKP",  # route 4
                      "MMX_SKP",  # route 5
                      "OTP_CRL",  # route 6
                      "SKP_MLH",  # route 7
                      "SKP_MMX"]  # route 8
        # load training datasets
        self.X_train = np.load('inputClf/X_train.npy')
        self.y_train = np.load('inputClf/y_train.npy')
        self.y_train_price = np.load('inputClf/y_train_price.npy')

        # deal with unbalanced data
        #self.X_train, self.y_train = self.dealingUnbalancedData(self.X_train, self.y_train)

        # load test datasets
        self.X_test = np.load('inputClf/X_test.npy')
        self.y_test = np.load('inputClf/y_test.npy')
        self.y_test_price = np.load('inputClf/y_test_price.npy')
        self.y_pred = np.empty(shape=(self.y_test.shape[0],1))


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

    def dealingUnbalancedData(self, X_train, y_train):
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
        # feature 4: prediction(buy or wait).
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


