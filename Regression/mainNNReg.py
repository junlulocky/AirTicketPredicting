# system library
import numpy as np

# user-library
from RegressionNN import RegressionNN



def mainNNReg():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test
    isNN = 1 # indicate it is neural network

    reg = RegressionNN(isTrain,isNN)
    reg.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    reg = RegressionRandomForest(isTrain)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """


if __name__ == "__main__":
    mainNNReg()