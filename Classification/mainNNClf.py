# system library
import numpy as np

# user-library
from ClassificationNN import ClassificationNN


def mainNNClf():
    nn = ClassificationNN(0) # 1 for train, 0 for test
    for i in range(8):
        print "Route: {}".format(i)
        nn.evaluateOneRouteForMultipleTimes(nn.routes[i])
    #nn.visualizePrediction(nn.routes[3])

if __name__ == "__main__":
    mainNNClf()