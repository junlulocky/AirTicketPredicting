# system library
import numpy as np

# user-library
import ClassficationBase


# third-party library
from sklearn import svm



class ClassificationSVM(ClassficationBase.ClassificationBase):
    def __init__(self):
        """
        The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
        different decision boundaries. This can be a consequence of the following
        differences:
        - ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
          regular hinge loss.

        - ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
          reduction while ``SVC`` uses the One-vs-One multiclass reduction.
        :return:
        """
        super(ClassificationSVM, self).__init__()
        self.clf = svm.SVC() # define the SVM classifier

        C = 1.0  # SVM regularization parameter
        self.svc = svm.SVC(kernel='linear', C=C)
        self.rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        self.poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
        self.lin_svc = svm.LinearSVC(C=C)



    def priceTolerance(self, percent):
        """
        Due to the sparsity of the buy entry, we may tolerate some low price as buy entry.
        :param percent: the percentage of the entrices to be buy
        :return: X_train, y_train
        """
        X_train_final = np.empty(shape=(0, 14))
        for flightNum in range(8):
            # concatenate the buy or wait info to get the total datas
            X_train = np.concatenate((self.X_train, self.y_train_price, self.y_train), axis=1)

            # choose one route datas
            X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

            # remove dummy variables
            # feature 0: departure date;  feature 1: observed date state
            # feature 2: minimum price; feature 3: maximum price
            # feature 4: prediction(buy or wait).
            #X_train = X_train[:, 8:14]

            # group by the feature: departure date
            departureDates_train = np.unique(X_train[:, 8])

            # get the final datas, the observed data state should be from large to small(i.e. for time series)
            X_train_tmp = np.empty(shape=(0, X_train.shape[1]))
            for departureDate in departureDates_train:
                indexs = np.where(X_train[:, 8]==departureDate)[0]
                datas = X_train[indexs, :]
                data_len = datas.shape[0]
                datas = sorted(datas, key=lambda entry: entry[12])
                datas = np.array(datas)

                for (i, data) in enumerate(datas):
                    if i < data_len*percent:
                        datas[i, 13] = 1

                """ visualize the datasets
                print departureDate
                print datas[:,8:14]
                """

                X_train_tmp = np.concatenate([X_train_tmp, datas])

            X_train_final = np.concatenate([X_train_final, X_train_tmp])

        self.X_train = X_train_final[:, 0:12]
        self.y_train_price = X_train_final[:, 12]
        self.y_train = X_train_final[:, 13]
        return X_train_final[:, 0:12], X_train_final[:, 13]

    def training(self):
        # change the shape of y_train to (n_samples, ) using ravel().
        self.svc.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)
        return self.y_pred