import numpy as np
import pylab as pl
from sklearn import mixture

n_samples = 300

# generate random sample, two parent functions, randn = gaussian
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                np.random.randn(n_samples, 2) + np.array([20, 20])]

clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X_train)

x = np.linspace(-20.0, 30.0, 500) # num points is 3rd param, default 50
y = np.linspace(-20.0, 40.0, 500)
X, Y = np.meshgrid(x, y)
XX = np.c_[X.ravel(), Y.ravel()] # still learning, i think numpy.c_ takes the 
# two 1-D arrays and creates a 2-D array with those as columns
# so, XX is a list of x,y points forming the meshgrid
# print clf.eval(XX)
Z = np.log(-clf.eval(XX)[0]) # doesn't it already return log probability,
#Z = -clf.eval(XX)[0]

# so this is log twice?
Z = Z.reshape(X.shape)      

CS = pl.contour(X, Y, Z)
CB = pl.colorbar(CS, shrink=0.8, extend='both')
pl.scatter(X_train[:, 0], X_train[:, 1], .8)

pl.axis('tight')
pl.show()                           


