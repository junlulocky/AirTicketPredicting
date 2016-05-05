# system library
import numpy as np

# user-library
import ClassficationBase


# third-party library
from sklearn.linear_model import Perceptron



class ClassificationPLA(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationPLA, self).__init__(isTrain, isOutlierRemoval)

        # data preprocessing
        self.dataPreprocessing()

        # PLA object
        self.clf = Perceptron()





    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the K Nearest Neighbors model
        self.clf.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        # predict the test data
        self.y_pred = self.clf.predict(self.X_test)

        # print the error rate
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)