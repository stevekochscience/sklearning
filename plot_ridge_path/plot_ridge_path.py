# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# modified by Steve Koch
# License: BSD Style.

import numpy as np
import pylab as pl
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = range(10, 0, -1)
print y
#y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-15, -7, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X,y)
    coefs.append(clf.coef_)
    
ax = pl.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
pl.gcf().canvas.set_window_title('fig%0.1f' %1 ) # i'm not sure why/how this works
pl.xlabel('alpha')
pl.ylabel('weights')
pl.title('Ridge coefficients as a function of the regularization')
pl.axis('tight')
pl.show()

