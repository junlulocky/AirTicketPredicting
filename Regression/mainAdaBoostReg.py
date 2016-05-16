# system library
import numpy as np

# user-library
from RegressionAdaBoost import RegressionAdaBoost


def mainAdaBoostReg():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test

    reg = RegressionAdaBoost(isTrain)
    reg.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    reg = RegressionRandomForest(isTrain)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """

def mainHyperparameter():
    """
    Parameter tuning
    """
    reg = RegressionAdaBoost(1)
    reg.parameterChoosing()


def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainAdaBoostReg()



if __name__ == "__main__":
    isParameterTuning = 1 # 1 for parameter tuning, 0 for evaluate routes
    main(isParameterTuning)