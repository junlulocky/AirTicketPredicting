# Machine Learning for Air Ticket Predicting
This project is still ongoing.

##Instructions on the codes#
Any theory about this project, please refer to my report.

I implemented many kinds of classifiers and regressors on this project in python. 

The outline of files and directories is below.

###Classification
```
|-inputClf # the input for classification method
|-inputClf_GMMOutlierRemoval # the input for classification method with ourlier removal by EM
|-inputClf_KMeansOutlierRemoval # the input for classification method with outlier removal by K-Means
# Classification methods
|-ClassificationBase.py  # The base class of the classification objects
	|-ClassificationAdaBoost.py   # AdaBoost class  
	|-ClassificationDecisionTree.py  # Decision Tree class
	|-ClassificationKNN.py # K nearest neighbot class
	|-ClassificationLogReg.py  # logistic regression class
	|-ClssificationNN.py # neural networks class
	|-ClassificationSVM.py # SVM class
# Classification test 
|-mainAdaBoostClf.py
|-mainDecisionTreeClf.py
|-mainKNNClf.py
|-mainLogisticReg.py
|-mainNNClf.py
|-mainSVMClf.py
```

###Regression
```  
|-inputReg # input for regression methods
# Regression methods
|-RegressionBase.py # The base class of the regression objects
  |-RegressionAdaBoost.py # AdaBoost class
  |-RegressionBlending.py # Uniform Blending class
  |-RegressionDecisionTree.py # Decision Tree class
  |-RegressionKNN.py # K nearest neighbors class
  |-RegressionLinReg.py # linear regression class
  |-RegressionNN.py # neural networks class
# Regression test
|-mainAdaBoostReg.py
|-mainBlendReg.py
|-mainDecisionTreeReg.py
|-mainKNNReg.py
|-mainLinReg.py
|-mainNNReg.py
|-mainSVMReg.py
```

###Aritificial Intelligence
```
# Artificial Intelligence methods
|-inputQLearning # input for qlearning method
|-qlearn.py # q learning class
|-mainQLearning.py # test for qlearning

```
