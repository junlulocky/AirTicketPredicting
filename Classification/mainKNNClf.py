# system library
import numpy as np

# user-library
from ClassificationKNN import ClassificationKNN


def mainKNNClf():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test
    isOutlierRemoval = 0 # 1 for outlier removal, 0 otherwise

    clf = ClassificationKNN(isTrain, isOutlierRemoval)
    clf.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    clf = ClassificationLogReg(isTrain, isOutlierRemoval)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """

def mainHyperparameter():
    """
    Parameter tuning
    """
    clf = ClassificationKNN(1, 0)
    clf.parameterChoosing()


def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainKNNClf()

def mainDrawValidationCurve():
    """
    Draw validation curve
    """
    clf = ClassificationKNN(1, 0)
    clf.drawValidationCurve()


if __name__ == "__main__":
    isdrawValidation = 0 # 1 to draw the validation curve; 0 to do analysis
    if isdrawValidation:
        mainDrawValidationCurve()
    else:
        isParameterTuning=0 # 1 for parameter tuning, 0 for evaluate routes
        main(isParameterTuning)