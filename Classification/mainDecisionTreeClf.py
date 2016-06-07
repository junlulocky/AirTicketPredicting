# system library
import numpy as np

# user-library
from ClassificationDecisionTree import ClassificationDecisionTree



def mainDecisionTreeClf():
    """
    Evaluate routes
    """
    isTrain = 0 # 1 for train, 0 for test
    isOutlierRemoval = 0 # 1 for outlier removal, 0 otherwise

    clf = ClassificationDecisionTree(isTrain, isOutlierRemoval)
    clf.evaluateAllRroutes()


    """
    # You can also evaluate the routes separately.
    clf = ClassificationDecisionTree(isTrain, isOutlierRemoval)
    [perfor, normaPefor] = clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    clf.visualizePrediction(clf.routes[i])
    """

def mainHyperparameter():
    """
    Parameter tuning
    """
    clf = ClassificationDecisionTree(1, 0)
    clf.parameterChoosing()

def mainDrawValidationCurve():
    """
    Draw validation curve
    """
    clf = ClassificationDecisionTree(1, 0)
    clf.drawValidationCurve()



def main(isParameterTuning=0):
    if isParameterTuning:
        mainHyperparameter()
    else:
        mainDecisionTreeClf()



if __name__ == "__main__":
    isdrawValidation = 0 # 1 to draw the validation curve; 0 to do analysis
    if isdrawValidation:
        mainDrawValidationCurve()
    else:
        isParameterTuning=0 # 1 for parameter tuning, 0 for evaluate routes
        main(isParameterTuning)

