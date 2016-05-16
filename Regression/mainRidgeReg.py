# system library
import numpy as np

# user-library
from RegressionRidgeReg import RegressionRidgeReg



def mainRidge():
    """
    Evaluate routes
    """
    isTrain = 1 # 1 for train, 0 for test

    reg = RegressionRidgeReg(isTrain)
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
    reg = RegressionRidgeReg(1)
    reg.parameterChoosing()


def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainRidge()



if __name__ == "__main__":
    isParameterTuning = 0 # 1 for parameter tuning, 0 for evaluate routes
    main(isParameterTuning)