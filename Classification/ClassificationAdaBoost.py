# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt


class ClassificationAdaBoost(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval):
        super(ClassificationAdaBoost, self).__init__(isTrain, isOutlierRemoval)
        # data preprocessing
        self.dataPreprocessing()

        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=7,
            algorithm="SAMME.R")
        # self.dt_stump = DecisionTreeClassifier(max_depth=14)
        # self.ada = AdaBoostClassifier(
        #     base_estimator=self.dt_stump,
        #     learning_rate=1,
        #     n_estimators=50,
        #     algorithm="SAMME")



    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the Decision Tree model
        self.ada.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        self.y_pred = self.ada.predict(self.X_test)

        # print the error rate
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)

    def parameterChoosing(self):
        # Set the parameters by cross-validation

        dts = []
        dts.append(DecisionTreeClassifier(max_depth=5))
        dts.append(DecisionTreeClassifier(max_depth=7))
        dts.append(DecisionTreeClassifier(max_depth=9))
        dts.append(DecisionTreeClassifier(max_depth=11))
        dts.append(DecisionTreeClassifier(max_depth=12))
        dts.append(DecisionTreeClassifier(max_depth=14))
        dts.append(DecisionTreeClassifier(max_depth=15))
        dts.append(DecisionTreeClassifier(max_depth=17))
        dts.append(DecisionTreeClassifier(max_depth=19))
        dts.append(DecisionTreeClassifier(max_depth=21))
        dts.append(DecisionTreeClassifier(max_depth=22))
        dts.append(DecisionTreeClassifier(max_depth=24))
        dts.append(DecisionTreeClassifier(max_depth=26))
        dts.append(DecisionTreeClassifier(max_depth=27))
        dts.append(DecisionTreeClassifier(max_depth=31))
        dts.append(DecisionTreeClassifier(max_depth=33))
        dts.append(DecisionTreeClassifier(max_depth=35))
        dts.append(DecisionTreeClassifier(max_depth=37))
        dts.append(DecisionTreeClassifier(max_depth=39))
        dts.append(DecisionTreeClassifier(max_depth=41))
        dts.append(DecisionTreeClassifier(max_depth=43))
        dts.append(DecisionTreeClassifier(max_depth=45))
        dts.append(DecisionTreeClassifier(max_depth=47))
        dts.append(DecisionTreeClassifier(max_depth=49))
        dts.append(DecisionTreeClassifier(max_depth=50))

        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")

        tuned_parameters = [{'base_estimator': dts,
                             'n_estimators': range(5,60),
                             'learning_rate': [1, 2, 3],
                             'algorithm': ["SAMME.R", "SAMME"]
                             }
                            ]

        clf = GridSearchCV(AdaBoostClassifier(n_estimators=30), tuned_parameters, cv=5, scoring='precision_weighted')
        clf.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

        print "Best parameters set found on development set:\n"
        print clf.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "Detailed classification report:\n"
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print classification_report(y_true, y_pred)

    def drawValidationCurve(self):
        """
        To draw the validation curve
        :return:NA
        """
        X, y = self.X_train, self.y_train.ravel()
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        train_sizes = range(2,75)
        train_scores, valid_scores = validation_curve(self.ada, X, y, "n_estimators",
                                              train_sizes, cv=5)

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
                 label="Training Precision")
        plt.plot(train_sizes, valid_scores_mean, '*-', color="g",
                 label="Cross-validation Precision")

        plt.legend(loc="best")

        plt.xlabel('Estimators')
        plt.ylabel('Precision')
        plt.title('Validation Curve with AdaBoost-DecisionTree on the parameter of Estimators')
        plt.grid(True)
        plt.show()

