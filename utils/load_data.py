# system-library
import json
import os
import numpy as np

# user-library
import util as util

"""
# data prepare for the specific data set
"""
routes_specific = ["BCN_BUD",  # route 1
              "BUD_BCN",  # route 2
              "CRL_OTP",  # route 3
              "MLH_SKP",  # route 4
              "MMX_SKP",  # route 5
              "OTP_CRL",  # route 6
              "SKP_MLH",  # route 7
              "SKP_MMX"]  # route 8
# for currency change
currency_specific = [1,      # route 1 - Euro
                 0.0032, # route 2 - Hungarian Forint
                 1,      # route 3 - Euro
                 1,      # route 4 - Euro
                 0.12,   # route 5 - Swedish Krona
                 0.25,   # route 6 - Romanian Leu
                 0.018,  # route 7 - Macedonian Denar
                 0.018   # route 8 - Macedonian Denar
                 ]

"""
# data prepare for the general data set
"""
routes_general = ["BGY_OTP", # route 1
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

# for currency change
currency_general = [1,      # route 1 - Euro
                 0.0032, # route 2 - Hungarian Forint
                 1,      # route 3 - Euro
                 1,      # route 4 - Euro
                 1,      # route 5 - Euro
                 1,      # route 6 - Euro
                 0.25,   # route 7 - Romanian Leu
                 0.25,    # route 8 - Romanian Leu
                 0.25,   # route 9 - Romanian Leu
                 0.037,  # route 10 - Czech Republic Koruna
                 1,      # route 11 - Euro
                 0.23    # route 12 - Polish Zloty
                 ]



def is_not_nullprice(data):
    """
    used by the filter to filter out the null entries
    :param data: input data
    :return: true if it's not null, false if null
    """
    return data and data["MinimumPrice"] != None

def check_if_only_one_flightNum(datas):
    """
    check whether the datas only contain one flight number
    :param datas: input data
    :return: Ture if the datas only contain one flight number, False otherwise
    """
    kinds = []
    for data in datas:
        kinds += data["Flights"]

    flightNums = []
    for kind in kinds:
        flightNums.append(kind["FlightNumber"])

    if len(util.remove_duplicates(flightNums)) == 1:
        return True
    else:
        return False



def load_data_with_prefix_and_dataset(filePrefix="BCN_BUD", dataset="large data set"):
    currentDir = os.path.dirname(os.path.realpath(__file__))
    observeDatesDirs = os.listdir(currentDir + "/data/" + dataset) # path directory of each observed date in the dataset

    filePaths = [] # keep all the file paths start with "filePrefix"
    data_decoded = [] # keep all the schedules start with "filePrefix"

    for date in observeDatesDirs:
        currentPath = currentDir + "/data/" + dataset + "/" + date

        try:
            files = os.listdir(currentPath) # file names in currect date directory
            for file in files:
                try:
                    if filePrefix in file:
                        filePath = os.path.join(currentPath, file)
                        filePaths.append(filePath)

                        fp = open(filePath, 'r')
                        datas_with_specific_date = json.load(fp)
                        # add observed data
                        for data in datas_with_specific_date:
                            #"Date" is the departure date, "ObservedDate" is the observed date
                            data["ObservedDate"] = date.replace("-", "")
                            data["State"] = util.days_between(data["Date"], data["ObservedDate"]) - 1
                        data_decoded += datas_with_specific_date # do not use append function

                except:
                    print "Not a json file"
        except:
            print "Not a directory, MAC OS contains .DS_Store file."

    # filter the null entries
    data_decoded = filter(is_not_nullprice, data_decoded)

    return data_decoded


def load_data_with_daysBeforeTakeoff_and_sameFlightNum(days, filePrefix="BCN_BUD", dataset="large data set"):
    """
    Load data with same flight number and the same days before takeoff.
    i.e. same equivalence class
    But in out dataset, one route means one flight number.
    :param days: the days before takeoff
    :param filePrefix: choose which route
    :param dataset: choose from wchich dataset
    :return: data with same flight number and the same days before takeoff
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
    output = [data for data in datas if util.days_between(data["ObservedDate"], data["Date"]) == days]

    return output

def get_departure_len(filePrefix="BCN_BUD", dataset="large data set"):
    """
    So far, used in QLearning, return the total departure date length in the chosen dataset.
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)

    # get different departure data in the same flight number,
    # to compute the Q Values for such (flight number, departure date) pair.
    departureDates = []
    [departureDates.append(data["Date"]) for data in datas]
    departureDates = util.remove_duplicates(departureDates)

    return len(departureDates)


def load_data_with_departureIndex(departureIndex, filePrefix="BCN_BUD", dataset="large data set"):
    """
    Given the departureIndex, return the dataset with specific departure date in the chosen dataset.
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)

    # get different departure data in the same flight number,
    # to compute the Q Values for such (flight number, departure date) pair.
    departureDates = []
    [departureDates.append(data["Date"]) for data in datas]
    departureDates = util.remove_duplicates(departureDates)

    # choose the departure date by departureIndex
    departureDate = departureDates[departureIndex]
    print "Evaluating departure date " + departureDate + "..."

    """
    # remove duplicate observedDate-departureDate pair
    observedDates = []
    [observedDates.append(data["ObservedDate"]) for data in datas if data["Date"]==departureDate]
    observedDates = util.remove_duplicates(observedDates)
    states = len(observedDates)
    #print states
    """


    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    #states = []
    #[states.append(data["State"]) for data in specificDatas]
    #print max(states)

    return specificDatas

def load_data_with_departureDate(departureDate, filePrefix="BCN_BUD", dataset="large data set"):
    """
    Given the departureIndex, return the dataset with specific departure date in the chosen dataset.
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)

    print "Evaluating departure date " + departureDate + "..."

    """
    # remove duplicate observedDate-departureDate pair
    observedDates = []
    [observedDates.append(data["ObservedDate"]) for data in datas if data["Date"]==departureDate]
    observedDates = util.remove_duplicates(observedDates)
    states = len(observedDates)
    #print states
    """


    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    return specificDatas


def getMinimumPrice(datas):
    """
    Given the dataset, return the minimum price in the dataset
    :param datas: input dataset(in QLearning and Neural Nets, it should have same departure date)
    :return: minimum price in the dataset
    """
    minimumPrice = util.getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = util.getPrice(data["MinimumPrice"])
        minimumPrice = price if price<minimumPrice else minimumPrice
    minimumPrice = minimumPrice

    return minimumPrice

def getOptimalState(datas):
    """
    Given the dataset, return the state correspongding to minimum price in the dataset
    :param datas: input dataset(in QLearning and Neural Nets, it should have same departure date)
    :return: minimum price state in the dataset
    """
    optimalState = 0
    minimumPrice = util.getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = util.getPrice(data["MinimumPrice"])
        state = data["State"]
        optimalState = state if price<minimumPrice else optimalState
        minimumPrice = price if price<minimumPrice else minimumPrice

    return optimalState

def getMaximumPrice(datas):
    """
    Given the dataset, return the maximum price in the dataset
    :param datas: input dataset(in QLearning and Neural Nets, it should have same departure date)
    :return: maximum price in the dataset
    """
    maximumPrice = util.getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = util.getPrice(data["MinimumPrice"])
        maximumPrice = price if price>maximumPrice else maximumPrice

    return maximumPrice

def getChosenPrice(state, datas):
    """
    Given the state, i.e. the days before departure, and the dataset, return the price
    :param state: the days before departure
    :param datas: input dataset(in QLearning, it should have same departure date)
    :return: the chosen price
    """
    for data in datas:
        if data["State"] == state:
            return util.getPrice(data["MinimumPrice"])

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

"""
# step 1. The main data load function - for classification for specific dataset
"""
def load_for_classification_for_Specific(dataset="Specific", routes=routes_specific):
    """
    Load the data for classification
    :param dataset: dataset
    :return: X_train, y_train, X_test, y_test
    """
    isOneOptimalState = False
    # Construct the input data
    dim = routes.__len__() + 4
    X_train = np.empty(shape=(0, dim))
    y_train = np.empty(shape=(0,1))
    y_train_price = np.empty(shape=(0,1))
    X_test = np.empty(shape=(0,dim))
    y_test = np.empty(shape=(0,1))
    y_test_price = np.empty(shape=(0,1))

    for filePrefix in routes:
        datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
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

            # if isOneOptimalState:
            #     # Method 1: only 1 entry is buy
            #     optimalState = getOptimalState(specificDatas)
            #     if data["State"] == optimalState:
            #        y_i = [1]
            # else:
            #     # Method 2: multiple entries can be buy
            #     minPrice = getMinimumPrice(specificDatas)
            #     if util.getPrice(data["MinimumPrice"]) == minPrice:
            #         y_i = [1]

            #Method 2: multiple entries can be buy
            minPrice = getMinimumPrice(specificDatas)
            if util.getPrice(data["MinimumPrice"]) == minPrice:
                y_i = [1]


            # keep price info
            y_price = [util.getPrice(data["MinimumPrice"])]

            if int(departureDate) < 20160229 and int(departureDate) >= 20151129: # choose date between "20151129-20160229(20160115)" as training data
                X_train = np.concatenate((X_train, [x_i]), axis=0)
                y_train = np.concatenate((y_train, [y_i]), axis=0)
                y_train_price = np.concatenate((y_train_price, [y_price]), axis=0)
            elif int(departureDate) < 20160508 and int(departureDate) >= 20160229: # choose date before "20160508(20160220)" as test data
                X_test = np.concatenate((X_test, [x_i]), axis=0)
                y_test = np.concatenate((y_test, [y_i]), axis=0)
                y_test_price = np.concatenate((y_test_price, [y_price]), axis=0)
            else:
                pass

            # X_train = np.concatenate((X_train, [x_i]), axis=0)
            # y_train = np.concatenate((y_train, [y_i]), axis=0)
            # y_train_price = np.concatenate((y_train_price, [y_price]), axis=0)

        # end of for datas
    # end of for routes


    """
    remove duplicate rows for train
    """
    tmp_train = np.concatenate((X_train, y_train, y_train_price), axis=1)
    new_array = [tuple(row) for row in tmp_train]
    tmp_train = np.unique(new_array)

    # get the result
    X_train = tmp_train[:, 0:12]
    y_train = tmp_train[:, 12]
    y_train_price = tmp_train[:, 13]

    """
    remove duplicate rows for test
    """
    tmp_test = np.concatenate((X_test, y_test, y_test_price), axis=1)
    new_array = [tuple(row) for row in tmp_test]
    tmp_test = np.unique(new_array)

    # get the result
    X_test = tmp_test[:, 0:12]
    y_test = tmp_test[:, 12]
    y_test_price = tmp_test[:, 13]

    # save the result
    np.save('inputSpecificRaw/X_train', X_train)
    np.save('inputSpecificRaw/y_train', y_train)
    np.save('inputSpecificRaw/y_train_price', y_train_price)
    np.save('inputSpecificRaw/X_test', X_test)
    np.save('inputSpecificRaw/y_test', y_test)
    np.save('inputSpecificRaw/y_test_price', y_test_price)

    return X_train, y_train, X_test, y_test

"""
# step 1. The main data load function - for classification for the general dataset
"""
def load_for_classification_for_General(dataset="General", routes=routes_general):
    """
    Load the data for classification
    :param dataset: dataset
    :return: X_train, y_train, X_test, y_test
    """
    isOneOptimalState = False
    # Construct the input data
    dim = routes.__len__() + 4
    X_train = np.empty(shape=(0, dim))
    y_train = np.empty(shape=(0,1))
    y_train_price = np.empty(shape=(0,1))

    for filePrefix in routes:
        print filePrefix
        datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
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

            minPrice = getMinimumPrice(specificDatas)
            if util.getPrice(data["MinimumPrice"]) == minPrice:
                y_i = [1]


            # keep price info
            y_price = [util.getPrice(data["MinimumPrice"])]

            X_train = np.concatenate((X_train, [x_i]), axis=0)
            y_train = np.concatenate((y_train, [y_i]), axis=0)
            y_train_price = np.concatenate((y_train_price, [y_price]), axis=0)

        # end of for datas
    # end of for routes


    """
    remove duplicate rows
    """
    tmp = np.concatenate((X_train, y_train, y_train_price), axis=1)
    new_array = [tuple(row) for row in tmp]
    tmp = np.unique(new_array)

    # # get the result
    # X_train = tmp[:, 0:16]
    # y_train = tmp[:, 16]
    # y_train_price = tmp[:, 17]

    # save the result
    np.save('inputGeneralRaw/X_train', X_train)
    np.save('inputGeneralRaw/y_train', y_train)
    np.save('inputGeneralRaw/y_train_price', y_train_price)
    np.save('inputGeneralRaw/tmp', tmp)

    return X_train, y_train, y_train_price


"""
# step 2. price normalize for the classification input - for specific
"""
def priceNormalize_for_Specific(routes=routes_specific, currency=currency_specific):
    """
    Different routes have different units for the price, normalize it as Euro.
    :return: NA
    example: priceNormalize_for_General()
    """
    """
    Get the input specific clf data for the training data set
    """
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    X_train = np.load('inputSpecificRaw/X_train.npy')
    y_train = np.load('inputSpecificRaw/y_train.npy')
    y_train_price = np.load('inputSpecificRaw/y_train_price.npy')

    # normalize feature 10, feature 11, feature 13
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    matrixTrain = np.empty(shape=(0, evalMatrix_train.shape[1]))
    for i in range(len(routes)):
        evalMatrix = evalMatrix_train[np.where(evalMatrix_train[:, i]==1)[0], :]
        evalMatrix[:, 10] *= currency[i]
        evalMatrix[:, 11] *= currency[i]
        evalMatrix[:, 13] *= currency[i]
        matrixTrain = np.concatenate((matrixTrain, evalMatrix), axis=0)

    X_train = matrixTrain[:, 0:12]
    y_train = matrixTrain[:, 12]
    y_train_price = matrixTrain[:, 13]

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))


    np.save('inputSpecificClf/X_train', X_train)
    np.save('inputSpecificClf/y_train', y_train)
    np.save('inputSpecificClf/y_train_price', y_train_price)

    """
    Get the input specific clf data for the test data set
    """
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    X_test = np.load('inputSpecificRaw/X_test.npy')
    y_test = np.load('inputSpecificRaw/y_test.npy')
    y_test_price = np.load('inputSpecificRaw/y_test_price.npy')

    # normalize feature 10, feature 11, feature 13
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix_test = np.concatenate((X_test, y_test, y_test_price), axis=1)
    evalMatrix_test = evalMatrix_test[np.where(evalMatrix_test[:,8]>=20)[0], :]

    matrixTest = np.empty(shape=(0, evalMatrix_test.shape[1]))
    for i in range(len(routes)):
        evalMatrix = evalMatrix_test[np.where(evalMatrix_test[:, i]==1)[0], :]
        evalMatrix[:, 10] *= currency[i]
        evalMatrix[:, 11] *= currency[i]
        evalMatrix[:, 13] *= currency[i]
        matrixTest = np.concatenate((matrixTest, evalMatrix), axis=0)

    X_test = matrixTest[:, 0:12]
    y_test = matrixTest[:, 12]
    y_test_price = matrixTest[:, 13]

    y_test = y_test.reshape((y_test.shape[0], 1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0], 1))


    np.save('inputSpecificClf/X_test', X_test)
    np.save('inputSpecificClf/y_test', y_test)
    np.save('inputSpecificClf/y_test_price', y_test_price)

"""
# step 2. price normalize for the classification input - for general
"""
def priceNormalize_for_General(routes=routes_general, currency=currency_general):
    """
    Different routes have different units for the price, normalize it as Euro.
    :return: NA
    example: priceNormalize_for_General()
    """
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    X_train = np.load('inputGeneralRaw/X_train.npy')
    y_train = np.load('inputGeneralRaw/y_train.npy')
    y_train_price = np.load('inputGeneralRaw/y_train_price.npy')

    # normalize feature 14, feature 15, feature 17
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # fearure 16: prediction(buy or wait); feature 17: price
    evalMatrix_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    matrixTrain = np.empty(shape=(0, evalMatrix_train.shape[1]))
    for i in range(len(routes)):
        evalMatrix = evalMatrix_train[np.where(evalMatrix_train[:, i]==1)[0], :]
        evalMatrix[:, 14] *= currency[i]
        evalMatrix[:, 15] *= currency[i]
        evalMatrix[:, 17] *= currency[i]
        matrixTrain = np.concatenate((matrixTrain, evalMatrix), axis=0)

    X_train = matrixTrain[:, 0:16]
    y_train = matrixTrain[:, 16]
    y_train_price = matrixTrain[:, 17]

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))

    #self.X_train = np.concatenate((self.X_train, self.y_train_price), axis=1)
    #self.X_test = np.concatenate((self.X_test, self.y_test_price), axis=1)

    np.save('inputGeneralClf/X_train', X_train)
    np.save('inputGeneralClf/y_train', y_train)
    np.save('inputGeneralClf/y_train_price', y_train_price)

"""
# step 3. get the regression input and output from classification inputs - for specific
"""
def getRegressionOutput_for_SpecificTrain(routes=routes_specific):
    """
    Get the regression output formula from the classification datasets.
    :return: Save the regression datasets into inputGeneralReg
    """
    X_train = np.load('inputSpecificClf2/X_train.npy')
    y_train = np.load('inputSpecificClf2/y_train.npy')
    y_train_price = np.load('inputSpecificClf2/y_train_price.npy')

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight numbers
    # feature 8: departure date;  feature 9: observed date state
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: prediction(buy or wait); feature 13: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    """
    # define the variables needed to be changed
    """
    dim = 14
    idx_departureDate = 8
    idx_minimumPrice = 10
    idx_output = 12
    idx_currentPrice = 13

    # Construct train data
    X_tmp = np.empty(shape=(0, dim))
    for flightNum in range(len(routes)):

        # choose one route datas
        X_flightNum = X_train[np.where(X_train[:, flightNum]==1)[0], :]

        # group by the feature: departure date
        departureDates_train = np.unique(X_flightNum[:, idx_departureDate])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates_train:
            indexs = np.where(X_flightNum[:, idx_departureDate]==departureDate)[0]
            datas = X_flightNum[indexs, :]
            minPrice = min(datas[:, idx_minimumPrice]) # get the minimum price for the output
            datas[:, idx_output] = minPrice
            """
            print departureDate
            print minPrice
            print datas
            """
            X_tmp = np.concatenate((X_tmp, datas), axis=0)

    X_train = X_tmp[:, 0:idx_output]
    y_train = X_tmp[:, idx_output]
    y_train_price = X_tmp[:, idx_currentPrice]
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))

    # regression has one more feature than classification
    X_train = np.concatenate((X_train, y_train_price), axis=1)
    np.save('inputSpecificReg2/X_train', X_train)
    np.save('inputSpecificReg2/y_train', y_train)
    np.save('inputSpecificReg2/y_train_price', y_train_price)

def getRegressionOutput_for_SpecificTest(routes=routes_specific):
    """
    Get the regression output formula from the classification datasets.
    :return: Save the regression datasets into inputGeneralReg
    """
    X_test = np.load('inputSpecificClf2/X_test.npy')
    y_test = np.load('inputSpecificClf2/y_test.npy')
    y_test_price = np.load('inputSpecificClf2/y_test_price.npy')

    # concatenate the buy or wait info to get the total datas
    y_test = y_test.reshape((y_test.shape[0],1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0],1))

    # feature 0~7: flight numbers
    # feature 8: departure date;  feature 9: observed date state
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: prediction(buy or wait); feature 13: current price
    X_test = np.concatenate((X_test, y_test, y_test_price), axis=1)

    """
    # define the variables needed to be changed
    """
    dim = 14
    idx_departureDate = 8
    idx_minimumPrice = 10
    idx_output = 12
    idx_currentPrice = 13

    # Construct train data
    X_tmp = np.empty(shape=(0, dim))
    for flightNum in range(len(routes)):

        # choose one route datas
        X_flightNum = X_test[np.where(X_test[:, flightNum]==1)[0], :]

        # group by the feature: departure date
        departureDates_test = np.unique(X_flightNum[:, idx_departureDate])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates_test:
            indexs = np.where(X_flightNum[:, idx_departureDate]==departureDate)[0]
            datas = X_flightNum[indexs, :]
            minPrice = min(datas[:, idx_minimumPrice]) # get the minimum price for the output
            datas[:, idx_output] = minPrice
            """
            print departureDate
            print minPrice
            print datas
            """
            X_tmp = np.concatenate((X_tmp, datas), axis=0)

    X_test = X_tmp[:, 0:idx_output]
    y_test = X_tmp[:, idx_output]
    y_test_price = X_tmp[:, idx_currentPrice]
    y_test = y_test.reshape((y_test.shape[0], 1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0], 1))

    # regression has one more feature than classification
    X_test = np.concatenate((X_test, y_test_price), axis=1)
    np.save('inputSpecificReg2/X_test', X_test)
    np.save('inputSpecificReg2/y_test', y_test)
    np.save('inputSpecificReg2/y_test_price', y_test_price)


"""
# step 3. get the regression input and output from classification inputs
"""
def getRegressionOutput_for_General(routes=routes_general):
    """
    Get the regression output formula from the classification datasets.
    :return: Save the regression datasets into inputGeneralReg
    """
    X_train = np.load('inputGeneralClf/X_train.npy')
    y_train = np.load('inputGeneralClf/y_train.npy')
    y_train_price = np.load('inputGeneralClf/y_train_price.npy')

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~11: flight numbers
    # feature 12: departure date;  feature 3: observed date state
    # feature 14: minimum price; feature 15: maximum price
    # feature 16: prediction(buy or wait); feature 17: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    """
    # define the variables needed to be changed
    """
    dim = 18
    idx_departureDate = 12
    idx_minimumPrice = 14
    idx_output = 16
    idx_currentPrice = 17

    # Construct train data
    X_tmp = np.empty(shape=(0, dim))
    for flightNum in range(len(routes)):

        # choose one route datas
        X_flightNum = X_train[np.where(X_train[:, flightNum]==1)[0], :]

        # group by the feature: departure date
        departureDates_train = np.unique(X_flightNum[:, idx_departureDate])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates_train:
            indexs = np.where(X_flightNum[:, idx_departureDate]==departureDate)[0]
            datas = X_flightNum[indexs, :]
            minPrice = min(datas[:, idx_minimumPrice]) # get the minimum price for the output
            datas[:, idx_output] = minPrice
            """
            print departureDate
            print minPrice
            print datas
            """
            X_tmp = np.concatenate((X_tmp, datas), axis=0)

    X_train = X_tmp[:, 0:idx_output]
    y_train = X_tmp[:, idx_output]
    y_train_price = X_tmp[:, idx_currentPrice]
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))

    # regression has one more feature than classification
    X_train = np.concatenate((X_train, y_train_price), axis=1)
    np.save('inputGeneralReg/X_train', X_train)
    np.save('inputGeneralReg/y_train', y_train)
    np.save('inputGeneralReg/y_train_price', y_train_price)

"""
# step 4. visualize for classification - for specific
"""
def visualizeData_for_SpecificClassification(filePrefix, isTrain=True, routes=routes_specific):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeData_for_SpecificClassification(routes_specific[1], routes_specific)
    """
    if isTrain:
        X_train = np.load('inputClf_small/X_train.npy')
        y_train = np.load('inputClf_small/y_train.npy')
        y_train_price = np.load('inputClf_small/y_train_price.npy')
    else:
        X_train = np.load('inputClf_small/X_test.npy')
        y_train = np.load('inputClf_small/y_test.npy')
        y_train_price = np.load('inputClf_small/y_test_price.npy')

    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price; feature 3: maximum price
    # feature 4: prediction(buy or wait); feature 5:price
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

"""
# step 4. visualize for classification - for general
"""
def visualizeTrainData_for_GeneralClassification(filePrefix, routes):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeTrainData_for_General(routes_general[1], routes_general)
    """
    X_train = np.load('inputGeneralClf_small/X_train.npy')
    y_train = np.load('inputGeneralClf_small/y_train.npy')
    y_train_price = np.load('inputGeneralClf_small/y_train_price.npy')


    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # normalize feature 14, feature 15, feature 17
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # fearure 16: prediction(buy or wait); feature 17: price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price; feature 3: maximum price
    # feature 4: prediction(buy or wait); feature 5:price
    X_train = X_train[:, 12:18]

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

"""
# step 5. visualize for regression - for general
"""
def visualizeTrainData_for_GeneralRegression(filePrefix, routes):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeTrainData_for_General(routes_general[1], routes_general)
    """
    X_train = np.load('inputGeneralReg/X_train.npy')
    y_train = np.load('inputGeneralReg/y_train.npy')
    y_train_price = np.load('inputGeneralReg/y_train_price.npy')

    """
    define the variables to be changed
    """
    dim = 19
    idx_departureDate = 12


    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # fearure 16: current price;
    # feature 17: minimum price; feature 18: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price by now; feature 3: maximum price by now
    # feature 4: current price;
    # feature 5: minimum price; feature 6: current price
    X_train = X_train[:, 12:dim]

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

"""
# step 5. visualize for regression - for specific
"""
def visualizeTrainData_for_SpecificRegression(filePrefix, routes):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeTrainData_for_SpecificRegression(routes_general[1], routes_general)
    """
    X_train = np.load('Regression/inputReg/X_train.npy')
    y_train = np.load('Regression/inputReg/y_train.npy')
    y_train_price = np.load('Regression/inputReg/y_train_price.npy')

    X_train2 = np.load('Regression/inputReg/X_test.npy')
    y_train2 = np.load('Regression/inputReg/y_test.npy')
    y_train2_price = np.load('Regression/inputReg/y_test_price.npy')

    X_train = np.concatenate((X_train, X_train2), axis=0)
    y_train = np.concatenate((y_train, y_train2), axis=0)
    y_train_price = np.concatenate((y_train_price, y_train2_price), axis=0)

    """
    define the variables to be changed
    """
    dim = 15
    idx_departureDate = 8


    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: current price;
    # feature 13: minimum price; feature 14: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price by now; feature 3: maximum price by now
    # feature 4: current price;
    # feature 5: minimum price; feature 6: current price
    X_train = X_train[:, idx_departureDate:dim]

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

def testClf():
    X_train = np.load('inputSpecificClf/X_train.npy')
    y_train = np.load('inputSpecificClf/y_train.npy')
    y_train_price = np.load('inputSpecificClf/y_train_price.npy')

    X_test = np.load('inputSpecificClf/X_test.npy')
    y_test = np.load('inputSpecificClf/y_test.npy')
    y_test_price = np.load('inputSpecificClf/y_test_price.npy')

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    y_price = np.concatenate((y_train_price, y_test_price), axis=0)

    y_train = y[np.where(X[:,8]<=112)[0], :]
    y_test = y[np.where( (X[:,8]>112) & (X[:,8]<142))[0], :]

    y_train_price = y_price[np.where(X[:,8]<=112)[0], :]
    y_test_price = y_price[np.where((X[:,8]>112) & (X[:,8]<142))[0], :]

    X_train = X[np.where(X[:,8]<=112)[0], :]
    X_test = X[np.where((X[:,8]>112) & (X[:,8]<142))[0], :]

    # save the result
    np.save('inputSpecificClf2/X_train', X_train)
    np.save('inputSpecificClf2/y_train', y_train)
    np.save('inputSpecificClf2/y_train_price', y_train_price)
    np.save('inputSpecificClf2/X_test', X_test)
    np.save('inputSpecificClf2/y_test', y_test)
    np.save('inputSpecificClf2/y_test_price', y_test_price)

    print X_train.shape
    print y_train.shape
    print y_train_price.shape
    print X_test.shape
    print y_test.shape
    print y_test_price.shape

def getGeneralRoutesSmall():
    """
    get the general routes, make the departure date period the same as test data set
    :return:
    """
    # normalize feature 14, feature 15, feature 17
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # fearure 16: prediction(buy or wait); feature 17: price
    X_train = np.load('inputGeneralClf/X_train.npy')
    y_train = np.load('inputGeneralClf/y_train.npy')
    y_train_price = np.load('inputGeneralClf/y_train_price.npy')

    y_train = y_train[np.where((X_train[:,12]>=67) & (X_train[:,12]<=102))[0], :]
    y_train_price = y_train_price[np.where((X_train[:,12]>=67) & (X_train[:,12]<=102))[0], :]
    X_train = X_train[np.where((X_train[:,12]>=67) & (X_train[:,12]<=102))[0], :]

    np.save('inputGeneralClf_small/X_train', X_train)
    np.save('inputGeneralClf_small/y_train', y_train)
    np.save('inputGeneralClf_small/y_train_price', y_train_price)


if __name__ == "__main__":
    # load_for_classification('small data set', routes_general)
    # priceNormalize_for_General()
    #visualizeTrainData_for_GeneralClassification(routes_general[1], routes_general)
    #visualizeTrainData_for_GeneralRegression(routes_general[1], routes_general)
    #visualizeTrainData_for_GeneralClassification(routes_general[1], routes_general)
    #visualizeTrainData_for_SpecificRegression(routes_specific[1], routes_specific)

    """
    STEP 1: load raw data
    """
    #load_for_classification_for_Specific()
    #load_for_classification_for_General()

    """
    STEP 2: get the data for the classification problem
    """
    #priceNormalize_for_Specific()
    #priceNormalize_for_General()

    """
    STEP 3: get the data for the regression problem
    """
    #getRegressionOutput_for_SpecificTrain()
    #getRegressionOutput_for_SpecificTest()

    """
    STEP 4: visualize the data set for classification problem
    """
    isTrain = 0
    #visualizeData_for_SpecificClassification(routes_specific[1], isTrain, routes_specific)
    visualizeTrainData_for_GeneralClassification(routes_general[11], routes_general)
    #testClf()

    """
    STEP 5: visualize the data set, but you can do this step at the classification object
    """
    #visualizeTrainData_for_SpecificRegression(routes_general[1], routes_general)














