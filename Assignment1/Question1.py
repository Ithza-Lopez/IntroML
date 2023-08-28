import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

# define function to be minimized f()
def f(X_m, theta_m):
 # [your work!]
    theta_1 = theta_m[0]
    theta_2 = theta_m[1]

    x_1 = X_m.iloc[:,0]
    x_2 = X_m.iloc[:,1]
    y_bar = 0.4638 + theta_1*x_1 + theta_2*x_2
    return y_bar

def mean_squared_error(real_y, expect_y):
    mse_sqrd_error = np.mean((real_y - expect_y)**2)
 # [your work!]
    return mse_sqrd_error

def mse_gradient(y_m, X_m, theta_m):
 # [your work!]P
    y_pred = f(X_m, theta_m)
    gradient = np.dot(X.T, y_pred - y) / len(y)
    # J_theta = (1/(2*n))*np.sum()
    return gradient

 # Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

tolerance = 1e-6
step_size = 4e-1
# theta = # [your work!] np.array([0,0]) or np.array([4,4]
theta = np.array([0,0])
theta_prev = np.array([1,1])
# [your work!]

# #MAKE LINEAR REGRESSION OBJECT
# regr = linear_model.LinearRegression()
# #train model using training setys
# reg_fit = regr.fit(X_train, y_train.values)

# #make predictions on trainign set
# y_train_pred = regr.predict(X_train)

while np.linalg.norm(theta - theta_prev) > tolerance:
 # [your work!]) for (1.c)
    gradient = mse_gradient(y_train, X_train, theta)
    theta_prev = theta
    theta = theta - step_size * gradient

print(theta)

#  plt.xlabel('BMI')
#  plt.ylabel('Diabetes risk')
#  plt.scatter(X_train,y_train_pred, color = black, linewidth = 2 )
