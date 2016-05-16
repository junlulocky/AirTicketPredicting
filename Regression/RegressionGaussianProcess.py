# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import gaussian_process
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


class RegressionGaussianProcess(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionGaussianProcess, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create Gaussian process object
        self.gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the linear regression model
        self.gp.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred, sigma2_pred = self.gp.predict(self.X_test, eval_MSE=True)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
