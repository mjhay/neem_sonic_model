import numpy as np
import scipy as sp
from sklearn.kernel_approximation import RBFSampler
from params import *
n_resid_fea = 100
vel_gamma = 1.0

depths_tr_vel = RBFSampler(vel_gamma, n_resid_fea)
depths_tr_vel = depths_tr_vel.fit
