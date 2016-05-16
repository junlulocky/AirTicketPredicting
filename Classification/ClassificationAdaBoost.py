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


class ClassificationAdaBoost(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain):
        super(ClassificationAdaBoost, self).__init__(isTrain)
        # data preprocessing
        self.dataPreprocessing()

        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")



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

