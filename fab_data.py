import numpy as np
from params import *
from sklearn.kernel_approximation import RBFSampler,Nystroem
from sklearn.ensemble import RandomTreesEmbedding

fabric = np.genfromtxt(fabric_file,delimiter=' ', skip_header=True).astype('float32')

depths = fabric[:,0].reshape(fabric[:,0].size,1)
depths_sd = depths.std()
depths_mean = depths.mean()
depths_norm = (depths - depths_mean)/depths_sd

transformer = Nystroem(kernel='rbf',gamma=gamma,n_components=n_fea)
depths_tr = transformer.fit_transform(depths_norm).astype('float32')
n_fea = depths_tr.shape[1]
wais_cs = fabric[:,1:4]
depths_test = np.linspace(-1,1,1000).reshape(1000,1)
depths_test_tr = transformer.transform(depths_test).astype('float32')
