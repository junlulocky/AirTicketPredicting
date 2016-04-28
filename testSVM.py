

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
print(X_train.shape)
print(X_test.shape)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'],
                     'gamma': np.logspace(-4, 3, 30),
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]},
                     {'kernel': ['poly'],
                      'degree': [1, 2, 3, 4],
                      'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                      'coef0': np.logspace(-4, 3, 30)},
                    {'kernel': ['linear'],
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}]

scores = ['precision', 'recall']
print SVC(C=1).get_params().keys()

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='precision_weighted')
    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:\n"
    print clf.best_params_

    print "Grid scores on development set:\n"
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params)

    print "Detailed classification report:\n"
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)