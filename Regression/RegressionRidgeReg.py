# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error



class RegressionRidgeReg(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionRidgeReg, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.model = linear_model.Ridge(alpha = 24420.530945486549)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'alpha': np.logspace(-5,5)
                             }
                            ]


        reg = GridSearchCV(linear_model.Ridge(alpha = 0.5), tuned_parameters, cv=5, scoring='mean_squared_error')
        reg.fit(self.X_train, self.y_train)

        print "Best parameters set found on development set:\n"
        print reg.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in reg.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print reg.scorer_

        print "MSE for test data set:"
        y_true, y_pred = self.y_test, reg.predict(self.X_test)
        print mean_squared_error(y_pred, y_true)

    def training(self):
        # train the NN model
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.model.predict(self.X_test)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
