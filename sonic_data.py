import numpy as np
from params import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomTreesEmbedding
from scipy.interpolate import interp1d
sonic_file = "vels_with_corruption.csv"
char_time = (3.0*1e3)**-1
depths_vels = np.genfromtxt(sonic_file,delimiter=' ',skip_header=True).astype('float32')
vels_train = depths_vels
sqvels_train_nd = (vels_train*char_time)**2
