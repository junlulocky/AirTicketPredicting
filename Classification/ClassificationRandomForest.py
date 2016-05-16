# system library
import numpy as np

# user-library
import ClassficationBase


# third-party library
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report



class ClassificationRandomForest(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationRandomForest, self).__init__(isTrain, isOutlierRemoval)

        # data preprocessing
        self.dataPreprocessing()

        # define the random forest object
        # self.clf = RandomForestClassifier(max_features='sqrt', n_estimators=32, max_depth=58)
        self.clf = RandomForestClassifier(max_features='log2', n_estimators=20, max_depth=30)


    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'max_depth': range(20,60),
                             'n_estimators': range(10,40),
                             'max_features': ['sqrt', 'log2', None]
                             }
                            ]

        clf = GridSearchCV(RandomForestClassifier(n_estimators=30), tuned_parameters, cv=5, scoring='precision_weighted')
        clf.fit(self.X_train, self.y_train.ravel())

        print "Best parameters set found on development set:\n"
        print clf.best_params_

        print "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

        print "Detailed classification report:\n"
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print classification_report(y_true, y_pred)

    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the K Nearest Neighbors model
        self.clf.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        # predict the test data
        self.y_pred = self.clf.predict(self.X_test)

        # print the error rate
        print "total size: {}".format(self.y_test.shape[0])
        print "y_test: {}".format(np.sum(self.y_test))
        print "y_pred: {}".format(np.sum(self.y_pred))
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)