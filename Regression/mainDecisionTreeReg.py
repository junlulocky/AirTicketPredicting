# system library
import numpy as np

# user-library
from RegressionDecisionTree import RegressionDecisionTree


def mainDecisionTreeReg():
    """
    Evaluate routes
    """
    isTrain = 1 # 1 for train, 0 for test

    reg = RegressionDecisionTree(isTrain)
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
    reg = RegressionDecisionTree(1)
    reg.parameterChoosing()


def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainDecisionTreeReg()

def mainDrawValidationCurve():
    """
    Draw validation curve
    """
    reg = RegressionDecisionTree(1)
    reg.drawValidationCurve()



if __name__ == "__main__":
    isdrawValidation = 0 # 1 to draw the validation curve; 0 to do analysis
    if isdrawValidation:
        mainDrawValidationCurve()
    else:
        isParameterTuning = 0 # 1 for parameter tuning, 0 for evaluate routes
        main(isParameterTuning)