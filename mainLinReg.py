# system library
import numpy as np

# user-library
from RegressionLinReg import RegressionLinReg



def mainLinReg():
    reg = RegressionLinReg(1) # 1 for train, 0 for test
    for i in range(8):
        print "Route: {}".format(i)
        reg.evaluateOneRouteForMultipleTimes(reg.routes[i], 5)
    #reg.visualizePrediction(reg.routes[0])


if __name__ == "__main__":
    mainLinReg()