# system library
import numpy as np
import scipy.stats as sci

# user-library
import util

# third-party library
from hmmlearn.hmm import GaussianHMM


def mainTest():
    pass



class HmmClassifier():
    def __init__(self, referenceSeqs, inputSeq):
        self.referenceSeqs = referenceSeqs
        self.inputSeq = inputSeq

        # feel free to change this model
        self.model = GaussianHMM(n_components=2, covariance_type="full", n_iter=2000)

    def predict(self):
        probs = []
        for referenceSeq in self.referenceSeqs:
            #print "reference: {}".format(referenceSeq)
            self.model.fit(referenceSeq)
            hidden_states = self.model.predict(referenceSeq)
            prob = self.model.score(self.inputSeq)
            probs.append(prob)

        # return the index of the max prob
        return probs.index(max(probs))





if __name__ == "__main__":
    mainTest()

