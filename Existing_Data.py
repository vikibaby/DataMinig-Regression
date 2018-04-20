import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import isnan
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics


LEARNING_RATE = 0.0001
ERROR_MARGIN  = 0.1
#--
# compute_error()
# This function computes the sum-of-squares error for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  error (scalar)
#--
def compute_error( M, x, w, y ):
    error = 0;
    for j in range (M):
        y_hat = w[0]+w[1]*x[j]
        error += math.pow(y[j]-y_hat,2)
    error = 0.5 * error
    return error

#--
# compute_r2()
# This function computes R^2 for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  r2 (scalar)
#--
def compute_r2( M, x, w, y ):
    sum_of_square_for_error = 0
    sum_of_square_for_total = 0
    y_mean = np.mean( y )
    y_pre = [0 for i in range( M )]
    for j in range (M):
        y_pre[j] = w[0]+w[1]*x[j]
        sum_of_square_for_error += math.pow(y[j]-y_pre[j],2)
        sum_of_square_for_total += math.pow(y[j]-y_mean,2)
    r_2_score = 1.0-sum_of_square_for_error/sum_of_square_for_total
    return r_2_score


#--
# gradient_descent_2()
# this function solves linear regression with gradient descent for 2
# parameters.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameter values (of size 2)
#  y = list of target values
#  alpha = learning rate
# output:
#  w = updated list of parameter values
#---
def gradient_descent_2( M, x, w, y, alpha ):
    for j in range (M):
        error = y[j] - w[0] - w[1]*x[j]
        w[0] = w[0] + alpha * error * 1    * ( 1.0 / M )
        w[1] = w[1] + alpha * error * x[j] * ( 1.0 / M )
    return w

#--
# MAIN
#--
df = pd.read_csv("london-borough-profiles-jan2018.csv")
Male_life_expectancy=[]
feMale_life_expectancy=[]
lenth = len(df)
for i in range (len(df)):
    temp = df.iloc[i]
    try:
        float(temp[70])
        float(temp[71])
    except:
        print "find a wrong value"
    else:
        if isnan(float(temp[70]))!=True and isnan(float(temp[71]))!=True :
            Male_life_expectancy.append(float(temp[70]))
            feMale_life_expectancy.append(float(temp[71]))
plt.figure()
plt.plot(Male_life_expectancy, feMale_life_expectancy , 'bo')
plt.xlabel('age(men)')
plt.ylabel('age(women)')
plt.title('raw data')
plt.savefig( "Existing_Data_life_expectancy.png" )
plt.show()
plt.close()

x = np.array(Male_life_expectancy)
y = np.array(feMale_life_expectancy)
M = len(x)
#-run gradient descent to compute the regression equation
alpha   = LEARNING_RATE
epsilon = ERROR_MARGIN
# initialise weights with 0's
w     = [0 for i in range( 2 )]
y_pre = [0 for i in range( M )]
curr_error = 0
prev_error = compute_error( M, x, w, y )
num_iters = 0
converged = False
all_r2s = []
while( not converged ):
    # adjust weights using gradient descent
    w = gradient_descent_2( M, x, w, y, alpha )
    # compute error
    curr_error = compute_error( M, x, w, y )
    r2 = compute_r2( M, x, w, y )
    all_r2s.append( r2 )
    num_iters = num_iters + 1
    if ( num_iters % 1000 == 0 ):
        print( 'num_iters = %d  prev_error = %f  curr_error = %f  r2 = %f' % ( num_iters, prev_error, curr_error, r2 ))
    if ( prev_error - curr_error < epsilon ):
        converged = True
    else:
        prev_error = curr_error

print 'My regression equation: y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x'
print 'My r2 score =' + str( r2 )

plt.figure()
plt.plot(Male_life_expectancy, feMale_life_expectancy , 'bo')
plt.xlabel('age(men)')
plt.ylabel('age(women)')
for j in range( M ):
    y_pre[j] = w[0] + w[1] * x[j]
plt.plot( x, y_pre, 'r' )
plt.title( 'my regression solution' )
plt.savefig( 'Existing_Data_linreg.png' )
plt.show()
plt.close()

#-plot and save change in R^2
plt.figure()
plt.plot( all_r2s, 'b-' )
plt.title( 'r^2 score' )
plt.xlabel( 'iteration' )
plt.savefig('Existing_Data-r2-score.png' )
plt.show()
plt.close()
