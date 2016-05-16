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


if __name__ == "__main__":
    mainLogReg()
    #mainHyperParameter()