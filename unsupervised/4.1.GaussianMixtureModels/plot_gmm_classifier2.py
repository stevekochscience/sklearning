# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# License: BSD Style.
# SJK: Quick (naive) try with DPGMM, doesn't work...don't know why yet

import pylab as pl
import numpy as np
import matplotlib as mpl

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import DPGMM

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        
iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(iris.target, n_folds=4) # this is for convenience in
# splitting the dataset. I don't understand it yet, except that it helps
# give us 4 "folds" of the data, and we'll use one for testing I guess
# this helps: http://en.wikipedia.org/wiki/Cross-validation_(statistics)
# I can't do it to do besides 3 folds, maybe related to this
# https://github.com/scikit-learn/scikit-learn/issues/1618
# print skf        
train_index, test_index = next(iter(skf)) # i don't understand iterables yet
# print train_index, test_index

X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

#n_classes = len(np.unique(y_train))
n_classes = 6

# Try DPGMMs using different types of covariances
classifiers = dict((covar_type, DPGMM(n_components=n_classes, 
                    covariance_type=covar_type, alpha=10, n_iter=100))
                    for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

pl.figure(figsize=(3 * n_classifiers / 2, 6))
pl.subplots_adjust(bottom=0.01, top=0.95, hspace=0.15, wspace=.05,
                   left=.01, right=.99)
                   
for index, (name, classifier) in enumerate(classifiers.iteritems()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
#    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
#                                  for i in xrange(n_classes)])
    # i don't understand the X_train[y_train == i], i suppose it somehow selects
    # in X_train based on y_train. Maybe a numpy thing?
    
    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)
    
    h = pl.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)
    
    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        pl.scatter(data[:, 0], data[:, 1], 0.8, color=color)
    # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        pl.plot(data[:, 0], data[:, 1], 'x', color=color)
    
    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    pl.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)
    
    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    pl.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)
    
    pl.xticks(())
    pl.yticks(())
    pl.title(name)

pl.legend(loc='lower right', prop=dict(size=12))

pl.show()
