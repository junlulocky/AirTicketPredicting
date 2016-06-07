# system library
import numpy as np

# user-library
from RegressionRandomForest import RegressionRandomForest


def mainRandomForestReg():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test

    reg = RegressionRandomForest(isTrain)
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
    reg = RegressionRandomForest(1)
    reg.parameterChoosing()


def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainRandomForestReg()

def mainDrawValidationCurve():
    """
    Draw validation curve
    """
    reg = RegressionRandomForest(1)
    reg.drawValidationCurve_maxdepth()
    reg.drawValidationCurve_estimators()


if __name__ == "__main__":
    isdrawValidation = 1 # 1 to draw the validation curve; 0 to do analysis
    if isdrawValidation:
        mainDrawValidationCurve()
    else:
        isParameterTuning = 0 # 1 for parameter tuning, 0 for evaluate routes
        main(isParameterTuning)