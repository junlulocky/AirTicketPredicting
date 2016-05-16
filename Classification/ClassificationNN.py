# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet



class ClassificationNN(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain, isOutlierRemoval, isNN=1):
        super(ClassificationNN, self).__init__(isTrain, isOutlierRemoval, isNN=1)
        # data preprocessing
        self.dataPreprocessing()

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
        # normalize different currency units == already normalized!
        #self.priceNormalize()

        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        self.Standardization()



    def training(self):
        # train the NN model
        self.net1.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        y_pred_train = self.net1.predict(self.X_train)
        self.y_pred = self.net1.predict(self.X_test)

        # 1 for buy, 0 for wait
        median = np.median(y_pred_train)
        mean = np.mean(y_pred_train)
        self.y_pred[self.y_pred>=median] = 1  # change this threshold
        self.y_pred[self.y_pred<median] = 0

        print "Number of buy: {}".format(np.count_nonzero(self.y_pred))
        print "Number of wait: {}".format(np.count_nonzero(1-self.y_pred))
        #print np.concatenate((y_test, y_pred), axis=1)
        #print y_pred.T.tolist()[0]
        #print map(round, y_pred.T.tolist()[0])
        #print len(y_pred.T.tolist())

        # print the error rate
        self.y_pred = self.y_pred.reshape((self.y_pred.shape[0], 1))
        err = 1 - np.sum(self.y_test == self.y_pred) * 1.0 / self.y_pred.shape[0]
        print "Error rate: {}".format(err)

        return self.X_test, self.y_pred

