# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error



class RegressionDecisionTree(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionDecisionTree, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.model = DecisionTreeRegressor(max_depth=7, max_features=None)

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'max_features': ['sqrt', 'log2', None],
                             'max_depth': range(2,1000),
                             }
                            ]


        reg = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring='mean_squared_error')
        reg.fit(self.X_train, self.y_train)

        print "Best parameters set found on development set:\n"
        print reg.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in reg.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "MSE for test data set:\n"
        y_true, y_pred = self.y_test, reg.predict(self.X_test)
        print mean_squared_error(y_true, y_pred)

    def training(self):
        # train the linear regression model
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.model.predict(self.X_test)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
