# Simple example of a linear regression using data set representing
# the price of houses based on the size in square feet. Then plots
# the training data, test data, and linear fit.


from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

# # # # #
# MODEL #
# # # # #
x = np.array([1120, 3450, 1980, 3050, 3720, 5500, 3020, 4200, 5780]) # sizes of houses
y = np.array([112000, 152300, 210200, 223000, 260000, 320000, 340900, 368900, 446000]) # prices of houses

# easy, no-hassle way to store all the relevant linear regression data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# set up plot showing training set in black
plt.plot(x, y, 'ro', color = 'black') # plot arrays x and y (matching respective elements)
plt.ylabel('Price')
plt.xlabel('Size of House')
plt.axis([0, 6000, 0, 500000]) # set x axis then y axis range
plt.plot(x, x * slope + intercept, 'b') # plot linear regression line

# # # # # # 
# PREDICT #
# # # # # #
def predict(size):
    return size * slope + intercept

# plot new data set based on training set linear regression
newSizes = np.array([1500, 2000, 2220, 3850, 4820, 5600])
guessedPrices = []
for size in newSizes:
    guessedPrices.append(predict(size))
plt.plot(newSizes, guessedPrices, 'ro', color = 'red')

# blue dots for training set, red for data set
plt.show()