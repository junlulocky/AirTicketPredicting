# Machine Learning for Air Ticket Predicting

##Instructions on the codes#
Any theory about this project, please refer to my report. If you want to keep track of the result performance, please refer to the "Performance Record.xlsx" file.

I implemented many kinds of classifiers and regressors on this project in python.

And the features I used in classification and regression is described in the report.

The package I used during the project is described in requirements.txt


##Code Structure

###Classification
Use Classification to predict
```
|-inputClf_small # the input for classification method
|-inputClf_GMMOutlierRemoval # the input for classification method with ourlier removal by EM
|-inputClf_KMeansOutlierRemoval # the input for classification method with outlier removal by K-Means
# Classification methods
|-ClassificationBase.py  # The base class of the classification objects
	|-ClassificationAdaBoost.py   # AdaBoost class  
	|-ClassificationDecisionTree.py  # Decision Tree class
	|-ClassificationKNN.py # K nearest neighbot class
	|-ClassificationLinearBlend.py # linear blending class
	|-ClassificationLogReg.py  # logistic regression class
	|-ClssificationNN.py # neural networks class
	|-ClassificationPLA.py # perceptron learning algorithm class
	|-ClassificationRandomForest.py # random forest algorithm class
	|-ClassificationSVM.py # SVM class
	|-ClassificationUniformBlend.py # uniform blending algorithm class
# Classification test
|-mainAdaBoostClf.py
|-mainDecisionTreeClf.py
|-mainGeneralizeClf.py
|-mainKNNClf.py
|-mainLinearBlendClf.py
|-mainLogisticReg.py
|-mainNNClf.py
|-mainPLA.py
|-mainRandomForestClf.py
|-mainSVMClf.py
|-mainUniformBlendClf.py
```

###Regression
Use regression to predict.
```  
|-inputReg # input for regression methods
# Regression methods
|-RegressionBase.py # The base class of the regression objects
	|-RegressionAdaBoost.py # AdaBoost class
	|-RegressionDecisionTree.py # Decision Tree class
	|-RegressionGaussianProcess.py # gaussian process class
	|-RegressionKNN.py # K nearest neighbors class
	|-RegressionLinReg.py # linear regression class
	|-RegressionNN.py # neural networks class
	|-RegressionRandomForest.py # random forest class
	|-RegressionRidgeReg.py # ridge regression class
	|-RegressionUniformBlend.py # Uniform Blending class
# Regression test
|-mainAdaBoostReg.py
|-mainDecision.py
|-mainGaussianProcess.py
|-mainLinReg.py
|-mainNNReg.py
|-mainRandomForestReg.py
|-mainRidgeReg.py
|-mainUniformBlendReg.py
```

###Aritificial Intelligence
Use Artificial Intelligence to predict, here mainly Q-Learning.
```
# Artificial Intelligence methods
|-inputQLearning # input for qlearning method
|-qlearn.py # q learning class
|-mainQLearning.py # test for qlearning

```

###GeneralizeMode
It is used to generalize the methods to new routes.
```
|-HmmClassifier.py
|-mainHMM.py
```

###Others
```
|-utils # load data, log function, and utils
|-priceBehaviorAnalysis.py # analyze the price behavior of several routes
|-requirements.txt # package requirements
|-Performance Record.xlsx # record the performance of various parameters
```
