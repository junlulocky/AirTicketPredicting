# system library
import numpy as np

# user-library
import RegressionBase


# third-party library
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit



class RegressionNN(RegressionBase.RegressionBase):
    def __init__(self, isTrain, isNN):
        super(RegressionNN, self).__init__(isTrain, isNN)
        # data preprocessing
        #self.dataPreprocessing()

        self.net1 = NeuralNet(
                        layers=[  # three layers: one hidden layer
                            ('input', layers.InputLayer),
                            ('hidden', layers.DenseLayer),
                            #('hidden2', layers.DenseLayer),
                            #('hidden3', layers.DenseLayer),
                            ('output', layers.DenseLayer),
                            ],
                        # layer parameters:
                        input_shape=(None, 13),  # input dimension is 13
                        hidden_num_units=6,  # number of units in hidden layer
                        #hidden2_num_units=8,  # number of units in hidden layer
                        #hidden3_num_units=4,  # number of units in hidden layer
                        output_nonlinearity=None,  # output layer uses sigmoid function
                        output_num_units=1,  # output dimension is 1

                        # obejctive function
                        objective_loss_function = lasagne.objectives.squared_error,

                        # optimization method:
                        update=lasagne.updates.nesterov_momentum,
                        update_learning_rate=0.002,
                        update_momentum=0.4,

                        # use 25% as validation
                        train_split=TrainSplit(eval_size=0.2),

                        regression=True,  # flag to indicate we're dealing with regression problem
                        max_epochs=100,  # we want to train this many epochs
                        verbose=0,
                        )

    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train the NN model
        self.net1.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.net1.predict(self.X_test)
