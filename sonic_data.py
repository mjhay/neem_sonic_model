import numpy as np
from params import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomTreesEmbedding
from scipy.interpolate import interp1d
sonic_file = "vels_with_corruption.csv"
char_time = (3.0*1e3)**-1
depths_vels = np.genfromtxt(sonic_file,delimiter=' ',skip_header=True).astype('float32')
#depths_vels = depths_vels[depths_vels[:,0]>250,:]
#depths_vels = depthvel[:,0].reshape(depthvel[:,0].size,1)
vels_train = depths_vels#[:,1:4][:,np.newaxis]
#;ntrain = int(np.ceil(depths_vels[:,0].max() - depths_vels[:,0].min()))
#depths_train = np.linspace(depths_vels[:,0].min(),depths_vels[:,0].max(),ntrain)[:,np.newaxis]
#depths_at_vels = depths_vels[:,0][:,np.newaxis]
#interp_pvel =  interp1d(depths_at_vels[:,0],pvel[:,0])
#interp_sv =  interp1d(depths_at_vels[:,0],pvel[:,0])
#interp_sh =  interp1d(depths_at_vels[:,0],pvel[:,0])
#pvels_train = interp_pvel(depths_train)
#vels_test = np.hstack([pvel,np.zeros(pvel.shape),np.zeros(pvel.shape)])
sqvels_train_nd = (vels_train*char_time)**2


