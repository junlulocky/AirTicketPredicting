# system library
import numpy as np

# user-library
import ClassficationBase


# third-party library
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report



class ClassificationSVM(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        """
        The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
        different decision boundaries. This can be a consequence of the following
        differences:
        - ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
          regular hinge loss.

        - ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
          reduction while ``SVC`` uses the One-vs-One multiclass reduction.
        :return:
        """
        super(ClassificationSVM, self).__init__(isTrain, isOutlierRemoval)

        # data preprocessing
        self.dataPreprocessing()
        self.clf = svm.SVC() # define the SVM classifier

        C = 1.0  # SVM regularization parameter
        self.svc = svm.SVC(kernel='linear', C=C, max_iter=100000)
        self.rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        self.poly_svc = svm.SVC(kernel='poly', coef0=1, degree=3, C=C)
        self.lin_svc = svm.LinearSVC(C=C)




    def parameterChoosing(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'],
                             'gamma': np.logspace(-4, 3, 30),
                             'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]},
                             {'kernel': ['poly'],
                              'degree': [1, 2, 3, 4],
                              'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                              'coef0': np.logspace(-4, 3, 30)},
                            {'kernel': ['linear'],
                             'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}]

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='precision_weighted')
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
        self.svc.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        # predict the test data
        self.y_pred = self.svc.predict(self.X_test)