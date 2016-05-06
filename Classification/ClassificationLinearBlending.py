"""
This part is not finished by now. My apologies.
Please use at your own risk.
"""

# system library
import numpy as np
import json
import math

# user-library
import ClassficationBase


# third-party library
from sklearn import neighbors
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


class ClassificationLinearBlending(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval=0):
        super(ClassificationLinearBlending, self).__init__(isTrain, isOutlierRemoval)
        # data preprocessing
        self.dataPreprocessing()

        # create logistic regression object
        self.logreg = linear_model.LogisticRegression(tol=1e-6, penalty='l1', C=0.0010985411419875584)

        # create adaboost object
        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")

        # create knn object
        self.knn = neighbors.KNeighborsClassifier(6, weights='uniform')

        # create decision tree object
        self.decisiontree = DecisionTreeClassifier(max_depth=50)

        # create neural network object
        self.net1 = NeuralNet(
                        layers=[  # three layers: one hidden layer
                            ('input', layers.InputLayer),
                            ('hidden', layers.DenseLayer),
                            #('hidden2', layers.DenseLayer),
                            ('output', layers.DenseLayer),
                            ],
                        # layer parameters:
                        input_shape=(None, 12),  # inut dimension is 12
                        hidden_num_units=6,  # number of units in hidden layer
                        #hidden2_num_units=3,  # number of units in hidden layer
                        output_nonlinearity=lasagne.nonlinearities.sigmoid,  # output layer uses sigmoid function
                        output_num_units=1,  # output dimension is 1

                        # optimization method:
                        update=nesterov_momentum,
                        update_learning_rate=0.002,
                        update_momentum=0.9,

                        regression=True,  # flag to indicate we're dealing with regression problem
                        max_epochs=25,  # we want to train this many epochs
                        verbose=0,
                        )



    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the models
        self.logreg.fit(self.X_train, self.y_train.ravel())
        self.ada.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))
        self.knn.fit(self.X_train, self.y_train.ravel())
        self.decisiontree.fit(self.X_train, self.y_train)
        self.net1.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        y_pred1 = self.logreg.predict(self.X_test)
        y_pred1 = y_pred1.reshape((y_pred1.shape[0], 1))

        y_pred2 = self.ada.predict(self.X_test)
        y_pred2 = y_pred2.reshape((y_pred2.shape[0], 1))

        y_pred3 = self.knn.predict(self.X_test)
        y_pred3 = y_pred3.reshape((y_pred3.shape[0], 1))

        y_pred4 = self.decisiontree.predict(self.X_test)
        y_pred4 = y_pred4.reshape((y_pred4.shape[0], 1))

        # predict neural network
                # predict the test data
        y_pred_train = self.net1.predict(self.X_train)
        y_pred5 = self.net1.predict(self.X_test)

        # keep all the predictions
        y_preds = []
        y_preds.append(y_pred1)
        y_preds.append(y_pred2)
        y_preds.append(y_pred3)
        y_preds.append(y_pred4)
        y_preds.append(y_pred5)

        # 1 for buy, 0 for wait
        median = np.median(y_pred_train)
        mean = np.mean(y_pred_train)
        y_pred5[y_pred5>=median] = 1  # change this threshold
        y_pred5[y_pred5<median] = 0
        y_pred5 = y_pred5.reshape((y_pred5.shape[0], 1))



        # get the error rate
        self.y_pred = (y_pred2+y_pred3+y_pred4)/3
        self.y_pred[self.y_pred >= 0.5] = 1
        self.y_pred[self.y_pred < 0.5] = 0
        e1 = 1 - np.sum(self.y_test == y_pred1) * 1.0 / y_pred1.shape[0]
        e2 = 1 - np.sum(self.y_test == y_pred2) * 1.0 / y_pred2.shape[0]
        e3 = 1 - np.sum(self.y_test == y_pred3) * 1.0 / y_pred3.shape[0]
        e4 = 1 - np.sum(self.y_test == y_pred4) * 1.0 / y_pred4.shape[0]
        e5 = 1 - np.sum(self.y_test == y_pred5) * 1.0 / y_pred5.shape[0]
        e = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "e1 = {}".format(e1)
        print "e2 = {}".format(e2)
        print "e3 = {}".format(e3)
        print "e4 = {}".format(e4)
        print "e5 = {}".format(e5)
        print "Uniform error = {}".format(e)

        # keep all the error rates
        errors = []
        errors.append(e1)
        errors.append(e2)
        errors.append(e3)
        errors.append(e4)
        errors.append(e5)


        # totalAlpha = 0
        # self.y_pred = np.zeros(shape=self.y_pred.shape)
        # for i in range(5):
        #     if errors[i] <= 0.5:
        #         alpha = math.log(math.sqrt((1-errors[i])/errors[i]))
        #         totalAlpha += alpha
        #         self.y_pred = self.y_pred + alpha*y_preds[i]
        #
        # # predict the blending output
        # self.y_pred = self.y_pred / totalAlpha
        # self.y_pred[self.y_pred >= 0.5] = 1
        # self.y_pred[self.y_pred < 0.5] = 0
        alpha2 = math.log(math.sqrt((1-e2)/e2))
        alpha3 = math.log(math.sqrt((1-e3)/e3))
        alpha4 = math.log(math.sqrt((1-e4)/e4))
        self.y_pred = (alpha2*y_pred2 + alpha3*y_pred3 + alpha4*y_pred4) * 1.0 /(alpha2+alpha3+alpha4)







