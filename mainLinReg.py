# system library
import numpy as np

# user-library
from RegressionLinReg import RegressionLinReg



def mainLin():
    reg = RegressionLinReg()
    reg.evaluateOneRouteForMultipleTimes(reg.routes[7])
    #reg.visualizePrediction(reg.routes[2])


if __name__ == "__main__":
    mainLin()