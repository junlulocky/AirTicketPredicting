# system library
import numpy as np

# user-library
from ClassificationKNN import ClassificationKNN


def mainKNNClf():
    nn = ClassificationKNN(1) # 1 for train, 0 for test
    for i in range(8):
        print "Route: {}".format(i)
        nn.evaluateOneRouteForMultipleTimes(nn.routes[i])
    #nn.visualizePrediction(nn.routes[0])

if __name__ == "__main__":
    mainKNNClf()