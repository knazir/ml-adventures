# Uses a data set with two categories (0 or 1). The data set could
# represent clients for a bank and whether or not they paid off their
# loan based on their bank balance (other examples work). Uses a
# logistic regression (sigmoid fit) to be able to predict whether
# a potential client will pay off their loan where if the probability
# that they will is >= 0.5, then they are put in the 1 category.
#
# Program plots the training data, test data, and sigmoid fit function.

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# p i = 1 / 1 + exp[ - ( b0 + b1 * x )]

# initialize training data set
x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

plt.plot(x1, y1, 'ro', color = 'blue')
plt.plot(x2, y2, 'ro', color = 'red')

classifier = LogisticRegression()
classifier.fit(X, y) # fits training set to logistic regression classifier

# calculates b0 and b1 using classifier
def model(classifier, x):
    return 1 / ( 1 + np.exp(- (classifier.intercept_ + classifier.coef_ * x )))

# plot sigmoid fit function
for i in range (1, 120, 1):
    plt.plot(i / 10.0 - 2, model(classifier, i / 10.0 - 2), 'ro', color = 'green')

# setup graph view
plt.axis([-2, 10, -0.5, 2])
plt.show()