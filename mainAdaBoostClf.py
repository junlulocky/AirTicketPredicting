# system library
import numpy as np

# user-library
from ClassificationAdaBoost import ClassificationAdaBoost


def mainDecisionTreeClf():
    clf = ClassificationAdaBoost(0) # 1 for train, 0 for test

    for i in range(8):
        print "Route: {}".format(i)
        clf.evaluateOneRouteForMultipleTimes(clf.routes[i])
    #clf.visualizePrediction(clf.routes[1])



if __name__ == "__main__":
    mainDecisionTreeClf()