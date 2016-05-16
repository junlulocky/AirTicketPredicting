# system library
import numpy as np

# user-library
from HmmClassifier import HmmClassifier


routes_specific = ["BCN_BUD",  # route 1
              "BUD_BCN",  # route 2
              "CRL_OTP",  # route 3
              "MLH_SKP",  # route 4
              "MMX_SKP",  # route 5
              "OTP_CRL",  # route 6
              "SKP_MLH",  # route 7
              "SKP_MMX"]  # route 8


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

def getFeaturesForGeneralRoutes_Regression():
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # fearure 16: current price;
    # feature 17: minimum price; feature 18: current price
    X_general = np.load('inputGeneralReg/X_train.npy')
    y_general = np.load('inputGeneralReg/y_train.npy')
    y_general_price = np.load('inputGeneralReg/y_train_price.npy')

    """
    remove the non-relevant datas
    """
    latestDepartureDate = 102  # need to change
    y_general = y_general[np.where(X_general[:, 12]<=102)[0], :]
    y_general_price = y_general_price[np.where(X_general[:, 12]<=102)[0], :]
    X_general = X_general[np.where(X_general[:, 12]<=102)[0], :]

    latestState = 5  # need to change
    y_general = y_general[np.where(X_general[:, 13]>=2)[0], :]
    y_general_price = y_general_price[np.where(X_general[:, 13]>=2)[0], :]
    X_general = X_general[np.where(X_general[:, 13]>=2)[0], :]

    X_general = np.concatenate((X_general, y_general, y_general_price), axis=1)



    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: current price;
    # feature 13: minimum price; feature 14: current price
    X_specific = np.load('inputReg/X_train.npy')
    y_specific = np.load('inputReg/y_train.npy')
    y_specific_price = np.load('inputReg/y_train_price.npy')

    X_train2 = np.load('inputReg/X_test.npy')
    y_train2 = np.load('inputReg/y_test.npy')
    y_train2_price = np.load('inputReg/y_test_price.npy')

    X_specific = np.concatenate((X_specific, X_train2), axis=0)
    y_specific = np.concatenate((y_specific, y_train2), axis=0)
    y_specific_price = np.concatenate((y_specific_price, y_train2_price), axis=0)

    """
    define the specific patterns
    """
    patterns = [[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1]]

    # +3, 1 for y_result, 1 for y_result_price, 1 for flightNum
    X_result = np.empty(shape=(0, X_specific.shape[1]+3))

    # keep different specific routes
    routesSpecific = []
    for flightNum in range(len(routes_specific)):
        tmpSpecific = X_specific[np.where(X_specific[:, flightNum]==1)[0], :]
        routesSpecific.append(tmpSpecific)



    for flightNum in range(len(routes_general)):
        # choose one route datas
        tmpGeneral = X_general[np.where(X_general[:, flightNum]==1)[0], :]
        # feature 0: departure date;  feature 1: observed date state
        # feature 2: minimum price by now; feature 3: maximum price by now
        # feature 4: current price;
        tmpGeneral = tmpGeneral[:, 12:19]

        # group by the feature: departure date
        departureDates = np.unique(tmpGeneral[:, 0])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates:
            indexs = np.where(tmpGeneral[:, 0]==departureDate)[0]
            # get the datas of same departureDate
            datasByDate = tmpGeneral[indexs, :]
            """
            # sort by the observed date state, from large to small(i.e. for time series)
            """
            datasByDate = datasByDate[(10-datasByDate[:,1]).argsort()]

            # group by the feature: state
            states = np.unique(datasByDate[:,1])
            for state in states:
                maxState = max(states)
                #print "State: {}, MaxState: {}".format(state, maxState)
                datasByDateAndState = datasByDate[np.where((datasByDate[:, 1]>=state) & (datasByDate[:, 1]<=maxState))[0], :]
                referenceSeqs = []
                isNoUse = 0
                for ii in range(len(routes_specific)):
                    referenceSeq_i = routesSpecific[ii][np.where(routesSpecific[ii][:,8]==departureDate)[0],:]
                    referenceSeq_i = referenceSeq_i[np.where((referenceSeq_i[:,9]>=state) & (referenceSeq_i[:,9]<=maxState))[0], :]

                    referenceSeq_i = referenceSeq_i[:,12]

                    """ keep the seqeunce long enough"""
                    if referenceSeq_i.shape[0] < 31:
                        isNoUse = 1
                    else:
                        referenceSeq_i = np.array(referenceSeq_i)
                        referenceSeq_i = referenceSeq_i.reshape((referenceSeq_i.shape[0], 1))
                    referenceSeqs.append(referenceSeq_i)

                if isNoUse:
                    #print "no use"
                    idx = 0
                else:
                    inputSeq = datasByDateAndState[:,4]
                    inputSeq = inputSeq.reshape((inputSeq.shape[0], 1))
                    hmmClassifier = HmmClassifier(referenceSeqs, inputSeq)
                    #print "idx: {}".format(hmmClassifier.predict())
                    idx = hmmClassifier.predict()

                datasByDateAndState = datasByDate[np.where((datasByDate[:, 1]==state) )[0], :]

                flightIndex = np.array([flightNum]).reshape((1,1))
                X_i = np.concatenate((np.array(patterns[idx]).reshape(1,8),datasByDateAndState, flightIndex), axis=1)
                X_result = np.concatenate((X_result, X_i), axis=0)




    print X_result.shape
    y_result = X_result[:,13]
    y_result_price = X_result[:,14]
    y_index = X_result[:,15]
    X_result = X_result[:, 0:13]

    np.save('inputGeneralRegParsed/X_train', X_result)
    np.save('inputGeneralRegParsed/y_train', y_result)
    np.save('inputGeneralRegParsed/y_train_price', y_result_price)
    np.save('inputGeneralRegParsed/y_index', y_index)


    return X_result, y_result, y_result_price

def getFeaturesForGeneralRoutes_Classification():
    # feature 0~11: flight number dummy variables
    # feature 12: departure date; feature 13: observed date state;
    # feature 14: minimum price; feature 15: maximum price
    # feature 16: buy or wait; feature 17: current price
    X_general = np.load('inputGeneralClf_small/X_train.npy')
    y_general = np.load('inputGeneralClf_small/y_train.npy')
    y_general_price = np.load('inputGeneralClf_small/y_train_price.npy')

    y_general = y_general.reshape((y_general.shape[0], 1))
    y_general_price = y_general_price.reshape((y_general_price.shape[0], 1))

    """
    remove the non-relevant datas
    """
    # latestDepartureDate = 102  # need to change
    # y_general = y_general[np.where(X_general[:, 12]<=102)[0], :]
    # y_general_price = y_general_price[np.where(X_general[:, 12]<=102)[0], :]
    # X_general = X_general[np.where(X_general[:, 12]<=102)[0], :]
    #
    # latestState = 5  # need to change
    # y_general = y_general[np.where(X_general[:, 13]>=2)[0], :]
    # y_general_price = y_general_price[np.where(X_general[:, 13]>=2)[0], :]
    # X_general = X_general[np.where(X_general[:, 13]>=2)[0], :]

    X_general = np.concatenate((X_general, y_general, y_general_price), axis=1)



    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: buy or wait; feature 13: current price
    X_specific = np.load('inputClf_small/X_train.npy')
    y_specific = np.load('inputClf_small/y_train.npy')
    y_specific_price = np.load('inputClf_small/y_train_price.npy')
    X_specific = np.concatenate((X_specific, y_specific, y_specific_price), axis=1)

    """
    define the specific patterns
    """
    patterns = [[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1]]

    # +3, 1 for y_result, 1 for y_result_price, 1 for flightNum
    X_result = np.empty(shape=(0, 12+3))

    # keep different specific routes separately
    routesSpecific = []
    for flightNum in range(len(routes_specific)):
        tmpSpecific = X_specific[np.where(X_specific[:, flightNum]==1)[0], :]
        routesSpecific.append(tmpSpecific)


    for flightNum in range(len(routes_general)):
        # choose one route datas
        tmpGeneral = X_general[np.where(X_general[:, flightNum]==1)[0], :]
        # feature 0: departure date;  feature 1: observed date state
        # feature 2: minimum price by now; feature 3: maximum price by now
        # feature 4: output; feature 5: current price
        tmpGeneral = tmpGeneral[:, 12:18]

        # group by the feature: departure date
        departureDates = np.unique(tmpGeneral[:, 0])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates:
            indexs = np.where(tmpGeneral[:, 0]==departureDate)[0]
            # get the datas of same departureDate
            datasByDate = tmpGeneral[indexs, :]
            """
            # sort by the observed date state, from large to small(i.e. for time series)
            """
            #datasByDate = datasByDate[(10-datasByDate[:,1]).argsort()]

            # group by the feature: state
            states = np.unique(datasByDate[:,1])
            for state in states:
                maxState = max(states)
                #print "State: {}, MaxState: {}".format(state, maxState)
                datasByDateAndState = datasByDate[np.where((datasByDate[:, 1]>=state) & (datasByDate[:, 1]<=maxState))[0], :]
                referenceSeqs = []
                isNoUse = 0
                for ii in range(len(routes_specific)):
                    referenceSeq_i = routesSpecific[ii][np.where(routesSpecific[ii][:,8]==departureDate)[0],:]
                    referenceSeq_i = referenceSeq_i[np.where((referenceSeq_i[:,9]>=state) & (referenceSeq_i[:,9]<=maxState))[0], :]

                    referenceSeq_i = referenceSeq_i[:,13]

                    """ keep the seqeunce long enough"""
                    if referenceSeq_i.shape[0] < 31:
                        isNoUse = 1
                    else:
                        referenceSeq_i = np.array(referenceSeq_i)
                        referenceSeq_i = referenceSeq_i.reshape((referenceSeq_i.shape[0], 1))
                    referenceSeqs.append(referenceSeq_i)

                if isNoUse:
                    #print "no use"
                    idx = 0
                else:
                    inputSeq = datasByDateAndState[:,5]
                    inputSeq = inputSeq.reshape((inputSeq.shape[0], 1))
                    hmmClassifier = HmmClassifier(referenceSeqs, inputSeq)
                    #print "idx: {}".format(hmmClassifier.predict())
                    idx = hmmClassifier.predict()

                datasByDateAndState = datasByDate[np.where((datasByDate[:, 1]==state) )[0], :]
                datasByDateAndState = datasByDateAndState[0,:]
                datasByDateAndState = datasByDateAndState.reshape((1,datasByDateAndState.shape[0]))

                flightIndex = np.array([flightNum]).reshape((1,1))
                X_i = np.concatenate((np.array(patterns[idx]).reshape(1,8),datasByDateAndState, flightIndex), axis=1)
                X_result = np.concatenate((X_result, X_i), axis=0)



    y_result = X_result[:,12]
    y_result_price = X_result[:,13]
    y_index = X_result[:,14]
    X_result = X_result[:, 0:12]

    np.save('inputGeneralClf_HmmParsed/X_train', X_result)
    np.save('inputGeneralClf_HmmParsed/y_train', y_result)
    np.save('inputGeneralClf_HmmParsed/y_train_price', y_result_price)
    np.save('inputGeneralClf_HmmParsed/y_index', y_index)

    return X_result, y_result, y_result_price



if __name__ == "__main__":
    getFeaturesForGeneralRoutes_Classification()
