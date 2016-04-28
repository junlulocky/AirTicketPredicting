# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



class RegressionAdaBoost(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionAdaBoost, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create AdaBoost regression object
        decisionReg = DecisionTreeRegressor(max_depth=10)
        rng = np.random.RandomState(1)
        self.adaReg = AdaBoostRegressor(decisionReg,
                          n_estimators=400,
                          random_state=rng)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the linear regression model
        self.adaReg.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        self.y_pred = self.adaReg.predict(self.X_test)
