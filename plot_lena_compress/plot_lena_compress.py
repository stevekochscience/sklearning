# Code source: Gael Varoqueux
# Modified for documentation merge by Jaques Grobler
# License: BSD

import numpy as np
import pylab as pl
import scipy as sp

from sklearn import cluster

from sys import argv
script, n_clusters = argv
n_clusters = int(n_clusters)

np.random.seed(0)

try:
    lena = sp.lena()
except AttributeError:
    # Newer versions of scipy have lena in misc
    from scipy import misc
    lena = misc.lena()

X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
# (-1,1) tuple makes it a 2-D array, with one dimension having size one and
# the other dimension having a size of the total number of original pixels
# basically flattening the image

k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze() # get rid of dimensions size-1
labels = k_means.labels_ # the labels say which is the best classification
# of each original pixel value.
#print labels[22050:22070]
#print values

# create an array from labels and values
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape

vmin = lena.min()
vmax = lena.max()

# original lena
pl.figure('original', figsize = (3, 2.2))
pl.imshow(lena, cmap=pl.cm.gray, vmin=vmin, vmax=256)

ncs = ' ' + str(n_clusters)

# compressed lena
pl.figure('compressed' + ncs, figsize=(3, 2.2))
pl.imshow(lena_compressed, cmap=pl.cm.gray, vmin=vmin, vmax=vmax)

# equal bins lena
regular_values = np.linspace(0, 256, n_clusters + 1)
regular_labels = np.searchsorted(regular_values, lena) - 1
regular_values = .5 * (regular_values[1:] + regular_values[:-1]) # mean
# Oh crap I'm not good at slicing yet.  the above line is just a concise
# way of reducing the size of regular_values array by 1 element, and
# replacing each element with the midpoint of two points in the starting array
regular_lena = np.choose(regular_labels.ravel(), regular_values)
regular_lena.shape = lena.shape
pl.figure('equal bins' + ncs, figsize=(3, 2.2))
pl.imshow(regular_lena, cmap=pl.cm.gray, vmin=vmin, vmax=vmax)

# histogram
pl.figure('histogram' + ncs, figsize = (3,2.2))
pl.clf()
pl.axes([.01, .01, .98, .98])
pl.hist(X, bins=256, color='.5', edgecolor='.5')
pl.yticks(())
pl.xticks(regular_values)
values=np.sort(values)
for center_1, center_2 in zip(values[:-1], values[1:]):
    pl.axvline(.5 * (center_1 + center_2), color='b')
    
for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
    pl.axvline(.5 * (center_1 + center_2), color='b', linestyle='--')
    
pl.show()
