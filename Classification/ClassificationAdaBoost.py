# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class ClassificationAdaBoost(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain):
        super(ClassificationAdaBoost, self).__init__(isTrain)
        # data preprocessing
        self.dataPreprocessing()

        self.dt_stump = DecisionTreeClassifier(max_depth=10)
        self.ada = AdaBoostClassifier(
            base_estimator=self.dt_stump,
            learning_rate=1,
            n_estimators=5,
            algorithm="SAMME.R")



    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the Decision Tree model
        self.ada.fit(self.X_train, self.y_train.reshape((self.y_train.shape[0], )))

    def predict(self):
        # predict the test data
        self.y_pred = self.ada.predict(self.X_test)

