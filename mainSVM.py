# system library
import numpy as np

# user-library
from ClassificationSVM import ClassificationSVM





def mainSVM():
    svm = ClassificationSVM()
    print np.count_nonzero(svm.y_train)
    svm.priceTolerance(0.502)
    #svm.visualizeTrainData(svm.routes[0])
    print np.count_nonzero(svm.y_train)
    svm.training()
    y_pred = svm.predict()

    y_pred = y_pred.reshape((y_pred.shape[0], 1))

    svm.visualizePrediction(svm.routes[1])
    print y_pred.shape
    print np.count_nonzero(y_pred)



if __name__ == "__main__":
    mainSVM()