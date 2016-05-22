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
from sklearn import linear_model
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



class RegressionUniformBlending(RegressionBase.RegressionBase):
    def __init__(self, isTrain):
        super(RegressionUniformBlending, self).__init__(isTrain)
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

        # Create linear regression object
        self.linRegr = linear_model.LinearRegression()

        # Create KNN regression object
        self.knn = neighbors.KNeighborsRegressor(86, weights='distance')

        # Create Decision Tree regression object
        self.decisionTree = DecisionTreeRegressor(max_depth=7, max_features=None)

        # Create AdaBoost regression object
        decisionReg = DecisionTreeRegressor(max_depth=10)
        rng = np.random.RandomState(1)
        self.adaReg = AdaBoostRegressor(decisionReg,
                          n_estimators=400,
                          random_state=rng)

        # Create linear regression object
        self.model = RandomForestRegressor(max_features='sqrt', n_estimators=32, max_depth=39)


    def dataPreprocessing(self):
        # due to the observation, standization does not help the optimization.
        # So do not use it!
        #self.Standardization()
        pass

    def training(self):
        # train each regression model
        self.net1.fit(self.X_train, self.y_train)
        self.linRegr.fit(self.X_train, self.y_train)
        self.knn.fit(self.X_train, self.y_train)
        self.decisionTree.fit(self.X_train, self.y_train)
        self.adaReg.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))


    def predict(self):
        # predict the test data
        y_pred1 = self.net1.predict(self.X_test)
        y_pred1 = y_pred1.reshape((y_pred1.shape[0], 1))

        y_pred2 = self.linRegr.predict(self.X_test)
        y_pred2 = y_pred2.reshape((y_pred2.shape[0], 1))

        y_pred3 = self.knn.predict(self.X_test)
        y_pred3 = y_pred3.reshape((y_pred3.shape[0], 1))

        y_pred4 = self.decisionTree.predict(self.X_test)
        y_pred4 = y_pred4.reshape((y_pred4.shape[0], 1))

        y_pred5 = self.adaReg.predict(self.X_test)
        y_pred5 = y_pred5.reshape((y_pred5.shape[0], 1))

        self.y_pred = (y_pred1+y_pred2+y_pred3+y_pred4+y_pred5)/5

        # print MSE
        mse = mean_squared_error(self.y_pred, self.y_test)
        print "MSE: {}".format(mse)




