# system library
import numpy as np

# user-library
from ClassificationKNN import ClassificationKNN


def mainKNNClf():
    nn = ClassificationKNN()
    nn.evaluateOneRouteForMultipleTimes(nn.routes[7])
    #nn.visualizePrediction(nn.routes[0])

if __name__ == "__main__":
    mainKNNClf()