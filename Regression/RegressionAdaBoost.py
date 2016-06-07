# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt



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

    def drawValidationCurve(self):
        """
        To draw the validation curve
        :return:NA
        """
        X, y = self.X_train, self.y_train.ravel()
        indices = np.arange(y.shape[0])
        #np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        train_sizes = range(5,700)
        train_scores, valid_scores = validation_curve(self.adaReg, X, y, "n_estimators",
                                              train_sizes, cv=5, scoring='mean_squared_error')
        train_scores = -1.0/5 *train_scores
        valid_scores = -1.0/5 *valid_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
        plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training MSE")
        plt.plot(train_sizes, valid_scores_mean, '*-', color="g",
                 label="Cross-validation MSE")

        plt.legend(loc="best")

        plt.xlabel('Estimators')
        plt.ylabel('MSE')
        plt.title('Validation Curve with AdaBoost-DecisionTree Regression\n on the parameter of Estimators when the Decsion Tree has max depth=10')
        plt.grid(True)
        plt.show()

    def training(self):
        # train the linear regression model
        self.adaReg.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        self.y_pred = self.adaReg.predict(self.X_test)

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
