# system library
import numpy as np

# user-library
from RegressionNN import RegressionNN



def mainNNReg():
    nn = RegressionNN(True) # true for train, false for test
    for i in range(8):
        print "Route: {}".format(i)
        nn.evaluateOneRouteForMultipleTimes(nn.routes[i])
    #nn.visualizePrediction(nn.routes[1])


if __name__ == "__main__":
    mainNNReg()