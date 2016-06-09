import numpy as np

"""
For the small data set - these datas are from the function in util
"""

normalizedPerformance = 0.6135  # the performance getting from AdaBoost-DecisionTree Classification

# random price list
randomPrices_test = [55.4820634921,
                          57.8067301587,
                          23.152037037,
                          33.3727319588,
                          35.3032044199,
                          41.1180555556,
                          56.3433402062,
                          60.2546519337]


# average predict price - predict by AdaBoost-DecisionTree Classification
avgPredPrice = [38.0852380952,
                40.6537142857,
                14.7122222222,
                26.7207692308,
                26.0030769231,
                18.5,
                36.2589230769,
                39.6512307692]

print np.mean(randomPrices_test)
print np.mean(randomPrices_test) - np.mean(avgPredPrice)

