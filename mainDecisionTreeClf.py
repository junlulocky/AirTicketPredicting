# system library
import numpy as np

# user-library
from ClassificationDecisionTree import ClassificationDecisionTree



def mainDecisionTreeClf():
    clf = ClassificationDecisionTree()

    for i in range(8):
        print "Route: {}".format(i)
        clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    #clf.visualizePrediction(clf.routes[1])



if __name__ == "__main__":
    mainDecisionTreeClf()