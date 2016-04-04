# system library
import numpy as np

# user-library
from RegressionKNN import RegressionKNN



def mainLinReg():
    reg = RegressionKNN(1) # 1 for train, 0 for test
    reg.evaluateOneRouteForMultipleTimes(reg.routes[7], 5)
    #reg.visualizePrediction(reg.routes[3])


if __name__ == "__main__":
    mainLinReg()