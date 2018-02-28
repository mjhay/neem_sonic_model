import numpy as np
import scipy as sp
import tensorflow as tf
from synth_eigs import logit
n_fea = 2000
gamma = 1e2
fabric_file = "synth_eigs_ts.csv"
sonic_file = "vels_with_corruption.csv"
sess = tf.Session()
class Whitener:
    def __init__(self,X):
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
    def whiten(self,Z):
        return (Z-self.Xmean)/(self.Xstd+1e-15)
    def unwhiten(self,Zw):
        return Zw*self.Xstd + self.Xmean
def init_weights(shape,stddev=1e-3):
    return tf.random_normal(shape, stddev=stddev)

def tf_logit(x):
    return tf.log(x) - tf.log(x[:,-1])[:,np.newaxis]

def a2_eig_loss(A2_logit,A2_obs_logit):
    return tf.reduce_mean(tf.pow(A2_logit-A2_obs_logit,2))

def X2a2(lam):
    return np.vstack([lam[:,0]**2,lam[:,1]**2,lam[:,2]**2,
    lam[:,1]*lam[:,2],lam[:,0]*lam[:,2],lam[:,0]*lam[:,1]]).T

def init_uninitialized():
    for var in tf.all_variables():
    	if not sess.run(tf.is_variable_initialized(var)):
	    sess.run(tf.variables_initializer([var]))

meps = np.loadtxt("../min_energy_900.txt").astype('float32')
meps = meps[meps[:,2]>=0,:]
a2_meps = X2a2(meps)
MEPdim = meps.shape[0]


neem_fabric = np.loadtxt(fabric_file).astype('float32')
a2_neem = np.hstack([neem_fabric[:,1:],np.zeros(neem_fabric[:,1:].shape)])
ts_depths = neem_fabric[:,0][:,np.newaxis]
logit_a2_neem = logit(a2_neem[:,0:3])
#depths_ph = tf.placeholder('float32',shape=(None,1),name='depthsph') #transformed depths
#W = tf.Variable(init_weights((n_fea,MEPdim),1e-6),name="W")
