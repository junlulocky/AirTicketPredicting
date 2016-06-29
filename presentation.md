# Representation Draft
Hello, my project title is "Machine Learning for predicting flight ticket prices". 

In this project, I solved two problems, one is called the **specific problem**, in which the historical datas of the flight routes are given; the other one is called the **generalized problem**, in which the historical data of some routes are not given. And I need to predict to buy or to wait for **every ticket query**.

## Work flow
This is the overall work flow of my project. In the upper of the figure, it shows the specific problem, in which we learn the formula from the training dataset, and use the formula to predict the test dataset.

In the bottom part of this figure, it shows the generalized problem, in which we extract the input feature from the samples itself and the test samples of the specific problem. And then used the learnt formula in specific problem to predict.

## Data collection
We collected 8 routes data from a major airplane company betwen Nov. 9, 2015 and Feb. 20, 2016 for the specific problem, **which will be split into training dataset and testing dataset.**

And we also collected 12 new routes data for generalized problem.

In our case, we split the datas between Nov. 9, 2015 and Jan. 15, 2016 as the training dataset; and the datas betweeen Jan. 16, 2016 and Feb. 20, 2016 as the testing dataset.

And Generalized problem dataset has the same period as the test dataset of specific problem. 

## Feature extraction
In this project, I extract 6 features, they are the flight number(encoded by dummy variables); the minimum price so far; the maximum price so far;

the  (number of days between the first query date(09.11.2015 in our case) and departure date), which is called query-to-departure;

the (number of days between the query date and departure date), which is called days-to-departure;

the current price;

**There are many other features can be extracted, for example indicator whether it is holiday or not, whether it is weekend or not. This may be tried for furture work.**

## Regression Methods
Firstly I tried regression methods for this problem. 

### Methodology
For the output of regression, I set it to be the minimum price for each departure date and each flight. 

As the output is the minimum price for each departure date and each flight. Our regression method is to predict the expected minimum price for a given departure date and given flight with the input features. As a result, if the current price is less than the expected the minimum price, we predict to buy; otherwise, we predict to wait.

### Methods and results
And I tried many regression methods to predict. All the hyperparameters are tuned from 5-fold cross validation. 

Random Forest Regression gets the best performance in regression method. 

However, it’s variance is not small enough, which means it is sensitive to different routes. In this case, although for some routes, it gets good performance, for other routes, it gets bad performance. From the aspect of the clients, it is not fair for the some clients to buy tickets for which the system may predict badly. 

So the preferred method in regression is AdaBoost-Decision Tree Regression method, which has smallest variance for 8 routes and a relative high performance. 

## Classification Methods

### Methodology
For the output, I set the data of which the price is the the minimum price from the query date to the departure date to 1(namely Class 1 - to buy), otherwise, we set it to be 0(namely Class 2 - to wait). 

As our classification method is to predict to buy or to wait with the input features. As a result, if the prediction is to buy, we buy the ticket, and we only buy the earliest one.

### Solving imbalanced dataset
Because the buy entry is very sparse, so the dataset is imbalanced. So I need to deal with the imbalanced dataset, otherwise the algorithm will tend to predict to wait.

And we choose random over sampling to deal with the imbalanced dataset, 

### Identification of outliers
I use K-means and mixture of gaussians to remove outliers.

**This approach was based on the fact that each characteristic class should be restricted to a certain volume of the input space. Consequently, we consider a sample to be an outlier if it does not belong to its labeled class cluster.**

In the case of K-Means, 5896 samples were tagged as outliers. In the case of EM algorithm, 7318 samples were tagged as outliers. 

To be more concrete, in the figure, the blue points in the volume of the green points should be considered as outliers, and vice-versa.

### Methods and results
In classifcation methods, And I also tried many algorithms to predict. All the hyperparameters are tuned from 5-fold cross validation. 

As we see, AdaBoost-DecisionTree, KNN, and Uniform Blending get positive performance for all the 8 routes and have smaller variance over these routes compared to other classification algorithms. 

The AdaBoost-DecisionTree method gets the best performance and a relative low variance over 8 routes. And as expected, the uniform blending method has the lowest variance just like the theory of uniform blending describes. 

## Q-learning 
Stanrard q-learning has this formula, with R(s, a) is the instant reward, γ is the discount factor for future rewards.

In our case, there are two actions in each state, one is buy, the other one is wait. And I set the award for buy to be the minus price of that state, which means the larger the better; 

and I set the award for wait to be maximum future award. In this sense, there is no fugure award for action buy, and there is no instant award for action wait.

And I also define an equivalence class, in which the data has same flight number and same days before takeoff.

Finally, our Q value is the average over the equivalence class.

### Result
As we see, the Q-Learning method has an acceptable performance and the variance is not large as well. The performance of it is a little worse than AdaBoost-DecisionTree Classification and Uniform blending Classification algorithms. 


## Generalized problem
**In this project, I also considered such a situation that some routes do not have any historical datas, in which case we cannot perform the learning algorithms at all. This situation is very common, because there are always some new routes to be added by the airplane company or the company may conceal the historical datas for some reasons. And also, this model can reduce computation time when we want to predict to buy or wait quickly because we do not need to train a large amount of training data.**

### HMM Sequence classification
Firstly, we tried the **HMM sequence classification**, which is also called **Maximum Likelihood classification** in automatic speech processing. 

**I then defined a equivalcen sequence, which have same first query date, same departure date, and same number of days before takeoff, which is shown in this figure.** 

As long as we have a new entry to predict, we extract the 8 equivalence sequences from the 8 speicic routes, and train the 8 routes by HMM model. Then we get 8 HMM models and compare the new sequence under these 8 HMM models to get the maximum likelihood of it. In other words, if the new sequnce get the maximum likelihood under HMM model 2, then we allocate the feature of the route 2 to the new entry. 

Finally, we use the learnt formula from the specific problem to predict. In my case, I used Adaboost-DecisionTree Classification to predict.

### Uniform blending
And I also tried the uniform blending algorithm for the generalized problem. The uniform blending algorithm is just to **allocate the 8 modelds**    to the new route respectively, and then average it.

### Result
As we can see, the uniform blending does not get any improvement. 

But the HMM Sequence Classification algorithm makes 9 routes get improvement, 3 routes have negative performance. 

Although the average performance is 31.71%, which is lower than that of the specific problem, it makes sense that we did not use any historical data of these routes to predict 

## Conclusion
For the specific problem and from the aspect of performance, AdaBoost-Decision Tree Classification is suggested to be the best model, which has 61.35% of optimal performance and has relatively small performance variance over the 8 different routes. 

From the aspect of performance variance for different routes, Uniform Blending Classification is chosen as the best model with relatively high performance. 

On the other hand, the Q-Learning method got a relatively high performance as well. 

Compare the validation curve of classification methods to that of regression methods, we could find that the cross validation error or precision of all the 5 folds in classification has far smaller variance than that of regression (shown in the report). We then considered that the classification model construction is more suitable in this problem. 

For the generalized problem(i.e. predict without the historical), we did not test many models. However, the HMM Sequence Classification based AdaBoost-Decision Tree Classification model got a good performance over 12 new routes, which has 31.71% of optimal performance. 

















