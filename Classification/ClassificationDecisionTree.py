# system library
import numpy as np
import json

# user-library
import ClassficationBase


# third-party library
from sklearn.tree import DecisionTreeClassifier


class ClassificationDecisionTree(ClassficationBase.ClassificationBase):
    def __init__(self, isTrain):
        super(ClassificationDecisionTree, self).__init__(isTrain)
        # data preprocessing
        self.dataPreprocessing()

        self.clf = DecisionTreeClassifier(max_depth=5)



    def dataPreprocessing(self):
        # deal with unbalanced data
        self.dealingUnbalancedData()

        # Standardization
        #self.Standardization()



    def training(self):
        # train the Decision Tree model
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        # predict the test data
        self.y_pred = self.clf.predict(self.X_test)

