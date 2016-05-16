# system library
import numpy as np

# user-library
import ClassficationBase


# third-party library
from sklearn import linear_model

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report



class ClassificationLogReg(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationLogReg, self).__init__(isTrain, isOutlierRemoval)

        # data preprocessing
        self.dataPreprocessing()

        # logistic regression object
        """
        Best parameters set found on development set:
            {'penalty': 'l1', 'C': 0.0010985411419875584}
        """
        self.clf = linear_model.LogisticRegression(tol=1e-6, penalty='l1', C=0.0010985411419875584)



    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'penalty': ['l1'],
                             'C': np.logspace(-5,5)},
                             {'penalty': ['l2'],
                              'C': np.logspace(-5,5)}]

        clf = GridSearchCV(linear_model.LogisticRegression(tol=1e-6), tuned_parameters, cv=5, scoring='precision_weighted')
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
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - (np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0])
        print "Error rate: {}".format(err)