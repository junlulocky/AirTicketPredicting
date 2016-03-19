# system library
import numpy as np

# user-library
from RegressionKNN import RegressionKNN



def mainLinReg():
    reg = RegressionKNN()
    reg.evaluateOneRouteForMultipleTimes(reg.routes[7], 5)
    #reg.visualizePrediction(reg.routes[1])


if __name__ == "__main__":
    mainLinReg()