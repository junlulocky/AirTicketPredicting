# system library
import numpy as np

# user-library
from RegressionNN import RegressionNN



def mainNN():
    nn = RegressionNN()
    #nn.visualizeTrainData(nn.routes[0])
    #nn.evaluateOneRouteForMultipleTimes(nn.routes[1])
    nn.evaluateOneRouteForMultipleTimes(nn.routes[2])
    nn.visualizePrediction(nn.routes[2])


if __name__ == "__main__":
    mainNN()