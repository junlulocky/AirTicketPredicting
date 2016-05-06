# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error



class RegressionRandomForest(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionRandomForest, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.model = RandomForestRegressor(max_features=None, n_estimators=21, max_depth=55)

    def parameterChoosing(self):
        #Set the parameters by cross-validation
        tuned_parameters = [{'max_depth': range(20,60),
                             'n_estimators': range(10,40),
                             'max_features': ['sqrt', 'log2', None]
                             }
                            ]

        clf = GridSearchCV(RandomForestRegressor(n_estimators=30), tuned_parameters, cv=5, scoring='mean_absolute_error')
        clf.fit(self.X_train, self.y_train.ravel())

        print "Best parameters set found on development set:\n"
        print clf.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "MSE for test data set:\n"
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print mean_absolute_error(y_true, y_pred)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the linear regression model
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        # predict the test data
        self.y_pred = self.model.predict(self.X_test)

        # print MSE
        print mean_absolute_error(self.y_test, self.y_pred)
