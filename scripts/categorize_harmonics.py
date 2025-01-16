# %%
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, whiten

# %%
# Set up logging
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)

# %%
root = os.path.normpath(os.path.join(os.getcwd(), '..', 'results'))
TIME_STEP = 620
path = os.path.join(root, f"metrics_{TIME_STEP:03d}.csv")
data = np.genfromtxt(path, delimiter=",", names=True)

print(data.dtype.names)

# %%
# this is basically 1 (except for the first eigenvalue)
#data['eigenvalues'] / (data['dirichlet_r'] + data['dirichlet_p'] + data['dirichlet_z'])

# %%
# This clustering works relatively well for the later time points, but it should probably be some kind of mixture model
k = 3
normalized_data = whiten(data['dirichlet_r'])
centroids, distortion = kmeans(normalized_data, k)
plt.plot(normalized_data, 'bo')
for i in range(k):
    plt.axhline(centroids[i])
plt.show()
