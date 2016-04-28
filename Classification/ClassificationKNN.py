# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn import neighbors


class ClassificationKNN(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationKNN, self).__init__(isTrain, isOutlierRemoval)
        # data preprocessing
        self.dataPreprocessing()

        # first parameter is the K neighbors
        # 'uniform' assigns uniform weights to each neighbor
        # 'distance' assigns weights proportional to the inverse of the distance from the query point
        # default metric is euclidean distance
        self.clf = neighbors.KNeighborsClassifier(6, weights='uniform')



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

