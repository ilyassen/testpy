import numpy as np

import linreg # import your code

num_train = 3674
num_test = 1224
path = "winequality/winequality-white.csv"

# load data matrices
X_train, Y_train, X_test, Y_test = linreg.load_data(path, num_train)

# theta = linreg.fit(X_train, Y_train)

# print("Fitted weights:")
# print(theta)

np.random.seed(0)
X_ass = np.random.randn(10, 5)
Y_ass = np.random.randn(10)
theta_ass = linreg.fit(X_ass, Y_ass)
print(np.abs(theta_ass[0] + 0.20))
assert np.abs(theta_ass[0] + 0.20) <= 1e-2, "Wrong value of theta!"