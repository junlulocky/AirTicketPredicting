# system library
import numpy as np

# user-library
from RegressionRandomForest import RegressionRandomForest



def mainRandomForestReg():
    isTrain = 1 # 1 for train, 0 for test

    performance = 0
    normalizedPerformance = 0
    priceTolerance = 5
    reg = RegressionRandomForest(isTrain)
    for i in range(8):
        print "Route: {}".format(i)
        [perfor, normaPefor] = reg.evaluateOneRouteForMultipleTimes(reg.routes[i], priceTolerance)
        performance += perfor
        normalizedPerformance += normaPefor

    performance = round(performance/8, 2)
    normalizedPerformance = round(normalizedPerformance/8, 2)


    print "\nAverage Performance: {}%".format(performance)
    print "Average Normalized Performance: {}%".format(normalizedPerformance)



if __name__ == "__main__":
    mainRandomForestReg()
    # reg = RegressionRandomForest(1)# 1 for train, 0 for test
    # reg.parameterChoosing()