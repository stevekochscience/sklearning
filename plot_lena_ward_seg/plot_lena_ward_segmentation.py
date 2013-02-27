# Author: Vincent Michel, 2010
#         Alexandre Gramfort, 2011
# License: BSD Style.

import time as time
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward

###
# Generate data
lena = sp.misc.lena()
#a = pl.imread("nachosgray.jpg")
a = pl.imread("nachosgray_crop.jpg")
lena = a # try the nachosimage instead of lena
print lena.shape
# Downsample the image by a factor of 4
lena = lena [::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
# this appears to just be a 2x2 simple average
X = np.reshape(lena, (-1, 1)) # At this point, flattening the image doesn't
# make sense to me
###
# Define the structure A of the data. Pixels connected to their neighbors.
# SJK: OHHH, now flattening makes sense.  Not sure how boundary conditions are
# dealt with
connectivity = grid_to_graph(*lena.shape) # can't remember what * means

###
# Compute clustering
print "Compute structured hierarchical clustering..."
st = time.time()
n_clusters = 5 # number of regions
ward = Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, lena.shape)
print "Elapsed time: ", time.time() - st
print "Number of pixels: ", label.size
print "Number of clusters: ", np.unique(label).size

###
# Plot the results on an image
pl.figure(figsize=(5, 5))
pl.imshow(lena, cmap=pl.cm.gray)
for l in range(n_clusters):
    pl.contour(label == l, contours=1,
               colors=[pl.cm.spectral(l / float(n_clusters)), ])
pl.xticks(())
pl.yticks(())
pl.show()

