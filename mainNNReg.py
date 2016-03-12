# system library
import numpy as np

# user-library
from RegressionNN import RegressionNN


def mainNN():
    nn = RegressionNN()
    #nn.visualizeTrainData(nn.routes[0])
    #nn.evaluateOneRouteForMultipleTimes(nn.routes[1])
    nn.evaluateOneRouteForMultipleTimes(nn.routes[0])


if __name__ == "__main__":
    mainNN()