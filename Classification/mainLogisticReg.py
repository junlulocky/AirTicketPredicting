# system library
import numpy as np

# user-library
from ClassificationLogReg import ClassificationLogReg

def mainHyperParameter():
    clf = ClassificationLogReg(0)
    clf.parameterChoosing()


def mainLogReg():
    isTrain = 0 # 1 for train, 0 for test
    isOutlierRemoval = 0 # 1 for outlier removal, 0 otherwise

    clf = ClassificationLogReg(isTrain, isOutlierRemoval)
    clf.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    clf = ClassificationLogReg(isTrain, isOutlierRemoval)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """
def mainDrawValidationCurve():
    """
    Draw validation curve
    """
    clf = ClassificationLogReg(1, 0)
    clf.drawValidationCurve()


if __name__ == "__main__":
    isdrawValidation = 1 # 1 to draw the validation curve; 0 to do analysis
    if isdrawValidation:
        mainDrawValidationCurve()
    else:
        mainLogReg()
        #mainHyperParameter()