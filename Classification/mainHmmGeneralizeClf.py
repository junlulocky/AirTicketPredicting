# system library
import numpy as np

# user-library
from ClassificationHmmGeneralize import ClassificationHmmGeneralize


def mainHmmGeneralClf():
    isTrain = 1 # 1 for train, 0 for test
    isOutlierRemoval = 1 # 1 for outlier removal, 0 otherwise

    performance = 0
    normalizedPerformance = 0
    clf = ClassificationHmmGeneralize(isTrain)

    normPerforms = []
    for i in range(12):
        print "Route: {}".format(i)
        [perfor, normaPefor] = clf.evaluateGeneral(clf.routes_general[i])
        normPerforms.append(normaPefor)
        performance += perfor
        normalizedPerformance += normaPefor

    performance = round(performance/8, 2)
    normalizedPerformance = round(normalizedPerformance/8, 2)


    print "\nAverage Performance: {}%".format(performance)
    print "Average Normalized Performance: {}%".format(normalizedPerformance)
    print "Normalized Performance Variance: {}".format(np.var(normPerforms))

if __name__ == "__main__":
    mainHmmGeneralClf()