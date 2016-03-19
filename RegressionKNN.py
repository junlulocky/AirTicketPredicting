# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import neighbors



class RegressionKNN(RegressionBase.RegressionBase):
    def __init__(self):
        super(RegressionKNN, self).__init__()
        # data preprocessing
        #self.dataPreprocessing()

        # Create KNN regression object
        # first parameter is the K neighbors
        # 'uniform' assigns uniform weights to each neighbor
        # 'distance' assigns weights proportional to the inverse of the distance from the query point
        # default metric is euclidean distance
        self.regr = neighbors.KNeighborsRegressor(6, weights='uniform')

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the linear regression model
        self.regr.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.regr.predict(self.X_test)
