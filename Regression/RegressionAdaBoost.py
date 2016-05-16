# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error



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

    def parameterChoosing(self):
        dts = []
        dts.append(DecisionTreeRegressor(max_depth=5, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=7, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=9, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=11, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=12, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=14, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=15, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=17, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=19, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=21, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=22, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=24, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=26, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=27, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=31, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=33, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=35, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=37, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=39, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=41, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=43, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=45, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=47, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=49, max_features='auto'))
        dts.append(DecisionTreeRegressor(max_depth=50, max_features='auto'))


        tuned_parameters = [{'base_estimator': dts,
                             'n_estimators': range(5,700),
                             'learning_rate': [1, 2, 3]
                             }
                            ]

        reg = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=5, scoring='mean_squared_error')
        reg.fit(self.X_train, self.y_train.ravel())

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
        self.adaReg.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        self.y_pred = self.adaReg.predict(self.X_test)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
