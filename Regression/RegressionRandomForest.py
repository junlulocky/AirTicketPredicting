# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt



class RegressionRandomForest(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionRandomForest, self).__init__(isTrain)
        # data preprocessing
        #self.dataPreprocessing()

        # Create linear regression object
        self.model = RandomForestRegressor(max_features='sqrt', n_estimators=32, max_depth=39)

    def parameterChoosing(self):
        #Set the parameters by cross-validation
        tuned_parameters = [{'max_depth': range(20,60),
                             'n_estimators': range(10,40),
                             'max_features': ['sqrt', 'log2', None]
                             }
                            ]

        clf = GridSearchCV(RandomForestRegressor(n_estimators=30), tuned_parameters, cv=5, scoring='mean_squared_error')
        clf.fit(self.X_train, self.y_train.ravel())

        print "Best parameters set found on development set:\n"
        print clf.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "MSE for test data set:\n"
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print mean_squared_error(y_true, y_pred)

    def drawValidationCurve_maxdepth(self):
        """
        To draw the validation curve
        :return:NA
        """
        X, y = self.X_train, self.y_train.ravel()
        indices = np.arange(y.shape[0])
        #np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        train_sizes = range(1,60)
        train_scores, valid_scores = validation_curve(self.model, X, y, "max_depth",
                                              train_sizes, cv=5, scoring='mean_squared_error')
        train_scores = -1.0/5*train_scores
        valid_scores = -1.0/5*valid_scores

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

        plt.xlabel('Max Depth')
        plt.ylabel('MSE')
        plt.title('Validation Curve with Random Forest Regression \non the parameter of Max Depth when n_estimators=32')
        plt.grid(True)
        plt.show()

    def drawValidationCurve_estimators(self):
        """
        To draw the validation curve
        :return:NA
        """
        X, y = self.X_train, self.y_train.ravel()
        indices = np.arange(y.shape[0])
        #np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        train_sizes = range(1,40)
        train_scores, valid_scores = validation_curve(self.model, X, y, "n_estimators",
                                              train_sizes, cv=5, scoring='mean_squared_error')
        train_scores = -1.0/5*train_scores
        valid_scores = -1.0/5*valid_scores

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
        plt.title('Validation Curve with Random Forest Regression \non the parameter of estimators when max_depth=39')
        plt.grid(True)
        plt.show()

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
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)
