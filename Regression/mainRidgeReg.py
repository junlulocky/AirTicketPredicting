# system library
import numpy as np

# user-library
from RegressionRidgeReg import RegressionRidgeReg



def mainRidge():
    reg = RegressionRidgeReg()
    reg.evaluateOneRouteForMultipleTimes(reg.routes[0])
    reg.visualizePrediction(reg.routes[0])


if __name__ == "__main__":
    mainRidge()