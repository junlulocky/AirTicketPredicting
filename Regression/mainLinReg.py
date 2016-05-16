# system library
import numpy as np

# user-library
from RegressionLinReg import RegressionLinReg



def mainLinReg():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test

    reg = RegressionLinReg(isTrain)
    reg.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    reg = RegressionRandomForest(isTrain)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """


if __name__ == "__main__":
    mainLinReg()