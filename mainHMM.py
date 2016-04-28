# system library
import numpy as np
import scipy.stats as sci

# user-library
import util

# third-party library
from hmmlearn.hmm import GaussianHMM


routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8


def loadOneRoute(filePrefix="BCN_BUD"):
    # load raw datas
    X_train = np.load('inputNN_NBuy/X_train.npy')
    y_train = np.load('inputNN_NBuy/y_train.npy')
    y_train_price = np.load('inputNN_NBuy/y_train_price.npy')
    X_test = np.load('inputNN_NBuy/X_test.npy')
    y_test = np.load('inputNN_NBuy/y_test.npy')
    y_test_price = np.load('inputNN_NBuy/y_test_price.npy')

    """
    y_price = np.empty(shape=(0, 1))
    for i in range(y_test_price.shape[0]):
        price = [util.getPrice(y_test_price[i, 0])]
        y_price = np.concatenate((y_price, [price]), axis=0)
    """

    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)
    X_test = np.concatenate((X_test, y_test, y_test_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]
    X_test = X_test[np.where(X_test[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price; feature 3: maximum price
    # feature 4: prediction(buy or wait).
    print "SHAPE: {}".format(X_train.shape)
    X_train = X_train[:, 8:14]
    X_test = X_test[:, 8:14]

    # group by the feature: departure date
    departureDates_train = np.unique(X_train[:, 0])
    departureDates_test = np.unique(X_test[:, 0])

    # get the final datas, the observed data state should be from large to small(i.e. for time series)
    X_train_final = np.empty(shape=(0, X_train.shape[1]))
    length_train = []
    for departureDate in departureDates_train:
        indexs = np.where(X_train[:, 0]==departureDate)[0]
        datas = X_train[indexs, :]
        length_train.append(len(datas))
        X_train_final = np.concatenate([X_train_final, datas])

    X_test_final = np.empty(shape=(0, X_test.shape[1]))
    length_test = []
    for departureDate in departureDates_test:
        indexs = np.where(X_test[:, 0]==departureDate)[0]
        datas = X_test[indexs, :]
        length_test.append(len(datas))
        """
        check the data again!
        print departureDate
        print datas
        """

        X_test_final = np.concatenate([X_test_final, datas])

    return X_train_final,length_train, X_test_final, length_test


def mainHMM(filePrefix):
    X_train,length_train, X_test, length_test = loadOneRoute(filePrefix)
    # Run Gaussian HMM
    print "fitting to HMM and decoding ..."
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=2000).fit(X_train[:, 0:5], length_train)
    hidden_states = model.predict(X_test[:, 0:5], length_test)
    print "done"

    # Print trained parameters and plot
    print("Transition matrix")
    print(model.transmat_)
    print("Start Prob")
    print(model.startprob_)

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))


    print np.array(hidden_states).reshape((sum(length_test), 1))



if __name__ == "__main__":
    mainHMM(routes[0])

