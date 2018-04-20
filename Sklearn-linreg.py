import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
#--
# MAIN
#--
x, y, p = datasets.make_regression( n_samples=1000, n_features=1, n_informative=1, noise=10, coef=True )
#-generate synthetic data set
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.10)
plt.plot(x_train, y_train , 'ro')
plt.title('train data')
plt.savefig('Sklearn_train_data.png')
plt.show()
plt.close()

plt.plot(x_test, y_test , 'bo')
regr = LinearRegression()
regr.fit(x_train,y_train)
pred = regr.predict(x_test)
plt.plot(x_test, pred, color='black', linewidth=2)
plt.title( 'scikit regression solution')
plt.savefig('Sklearn_regression.png')
plt.show()
plt.close()

print 'scikit regression equation: y = ',
print regr.intercept_,
print ' + ',
print regr.coef_[0],
print 'x'
print 'scikit r2 = ',
print metrics.r2_score( y_test, pred )
print 'scikit error = ',
print metrics.mean_squared_error( y_test, pred )
