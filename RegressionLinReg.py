# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import linear_model



class RegressionLinReg(RegressionBase.RegressionBase):
    def __init__(self):
        super(RegressionLinReg, self).__init__()
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.regr = linear_model.LinearRegression()

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the NN model
        self.regr.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.regr.predict(self.X_test)
