# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error



class RegressionKNN(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionKNN, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create KNN regression object
        # first parameter is the K neighbors
        # 'uniform' assigns uniform weights to each neighbor
        # 'distance' assigns weights proportional to the inverse of the distance from the query point
        # default metric is euclidean distance
        self.regr = neighbors.KNeighborsRegressor(86, weights='distance')

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'weights': ['uniform', 'distance'],
                             'n_neighbors': range(2,100)
                             }
                            ]


        reg = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, cv=5, scoring='mean_squared_error')
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
        # train the linear regression model
        self.regr.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.regr.predict(self.X_test)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
