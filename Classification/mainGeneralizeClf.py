# system library
import numpy as np

# user-library
from ClassificationGeneralize import ClassificationGeneralize


def mainGeneralClf():
    isTrain = 1 # 1 for train, 0 for test
    isOutlierRemoval = 1 # 1 for outlier removal, 0 otherwise

    performance = 0
    normalizedPerformance = 0
    clf = ClassificationGeneralize(isTrain)
    for i in range(12):
        print "Route: {}".format(i)
        [perfor, normaPefor] = clf.evaluateGeneral(clf.routes_general[i])
        performance += perfor
        normalizedPerformance += normaPefor

    performance = round(performance/8, 2)
    normalizedPerformance = round(normalizedPerformance/8, 2)


    print "\nAverage Performance: {}%".format(performance)
    print "Average Normalized Performance: {}%".format(normalizedPerformance)

if __name__ == "__main__":
    #mainGeneralClf()
    clf = ClassificationGeneralize(1)
    clf.getMinTicketPriceForAllRoutesByNumpy()
    print clf.minPrices_general

    clf.getRandomTicketPriceForAllRoutesByNumpy()
    print clf.randomPrices_general

    clf.evaluateGeneralOneRoute(clf.routes_general[0])