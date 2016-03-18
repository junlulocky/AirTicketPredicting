# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor



class RegressionDecisionTree(RegressionBase.RegressionBase):
    def __init__(self):
        super(RegressionDecisionTree, self).__init__()
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.model = DecisionTreeRegressor(max_depth=6)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the linear regression model
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.model.predict(self.X_test)
