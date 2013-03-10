# Code source: Jaques Grobler
# License BSD

import pylab as pl
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes() #described in
# http://scipy-lectures.github.com/advanced/scikit-learn/index.html

#i=0
#j=0
#for row in diabetes.data:
#    for element in row:
#        print i, j
#        j += 1
#    i += 1
#    j = 0    
     
diabetes_X = diabetes.data[:, np.newaxis] # adds a new axis after first axis,
#                                           of length 1
diabetes_X_temp = diabetes_X[:, :, 2] # removes 3rd axis, by selecting 3rd item
# I don't know what this column is, but it has a mean of zero(ish) and values
# between -.2 and +.2

diabetes_X_train = diabetes_X_temp[:-20] # all but last twenty
diabetes_X_test = diabetes_X_temp[-20:] # last twenty

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression() # create linear regression object

regr.fit(diabetes_X_train, diabetes_y_train)

print 'Coefficients: ', regr.coef_
print 'Intercept: ', regr.intercept_
print ("Residual sum of squares: %.2f" %
       np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print ('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

pl.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
        linewidth=3)

pl.xticks(())
pl.yticks(())

pl.show()


