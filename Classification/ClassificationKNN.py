# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt


class ClassificationKNN(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationKNN, self).__init__(isTrain, isOutlierRemoval)
        # data preprocessing
        self.dataPreprocessing()

        # first parameter is the K neighbors
        # 'uniform' assigns uniform weights to each neighbor
        # 'distance' assigns weights proportional to the inverse of the distance from the query point
        # default metric is euclidean distance
        self.clf = neighbors.KNeighborsClassifier(2, weights='uniform')



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
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)

    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'weights': ['uniform', 'distance'],
                             'n_neighbors': range(2,60)
                             }
                            ]


        clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=5, scoring='precision_weighted')
        clf.fit(self.X_train, self.y_train.ravel())

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

        train_sizes = range(1,60)
        train_scores, valid_scores = validation_curve(self.clf, X, y, "n_neighbors",
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

        plt.xlabel('K Neighbors')
        plt.ylabel('Precision')
        plt.title('Validation Curve with KNN on the parameter of K')
        plt.grid(True)
        plt.show()


