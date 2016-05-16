# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


class ClassificationDecisionTree(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval):
        super(ClassificationDecisionTree, self).__init__(isTrain, isOutlierRemoval)
        # data preprocessing
        self.dataPreprocessing()

        self.clf = DecisionTreeClassifier(max_depth=45, max_features='log2')



    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()

    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'max_depth': range(2,60),
                             'max_features': ['sqrt', 'log2', None]
                             }
                            ]

        clf = GridSearchCV(DecisionTreeClassifier(max_depth=5), tuned_parameters, cv=5, scoring='precision_weighted')
        clf.fit(self.X_train, self.y_train.ravel())

        print "Best parameters set found on development set:\n"
        print clf.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "Detailed classification report:\n"
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print classification_report(y_true, y_pred)



    def training(self):
        # train the Decision Tree model
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.clf.predict(self.X_test)

        # print the error rate
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)

