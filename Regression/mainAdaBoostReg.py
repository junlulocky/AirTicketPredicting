# system library
import numpy as np

# user-library
from RegressionAdaBoost import RegressionAdaBoost



def mainLinReg():
    reg = RegressionAdaBoost(0) # 1 for train, 0 for test

    performance = 0
    normalizedPerformance = 0
    for i in range(8):
        print "Route: {}".format(i)
        [perfor, normaPefor] = reg.evaluateOneRouteForMultipleTimes(reg.routes[i], 5)
        performance += perfor
        normalizedPerformance += normaPefor

    performance = round(performance/8, 2)
    normalizedPerformance = round(normalizedPerformance/8, 2)

    print "Average Performance: {}%".format(performance)
    print "Average Normalized Performance: {}%".format(normalizedPerformance)

    #reg.visualizePrediction(reg.routes[3])


if __name__ == "__main__":
    mainLinReg()