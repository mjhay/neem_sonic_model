from params import *
from synth_eigs import depths_train,depths_ts,logit
from sklearn.cluster import KMeans
import sklearn.gaussian_process as gp
from gp_extras.kernels import HeteroscedasticKernel

prototypes = KMeans(n_clusters=10).fit(ts_depths).cluster_centers_

kern=[0,0,0]
gpr=[0,0,0]
means_at_train = np.zeros((depths_train.shape[0],3))
eigs_ts = np.genfromtxt(fabric_file,delimiter=' ',skip_header=1)#np.zeros((ntrain,3))
sd_at_train = 0.1*np.ones((depths_train.shape[0],3))#np.genfromtxt(fabric_file,delimiter=' ',skip_header=1)#np.zeros((ntrain,3))
#sd_at_train = 0.0001#np.zeros((ntrain,3))
#for i in range(0,3):
#    kern[i] = HeteroscedasticKernel.construct(prototypes,sigma_2=1e2, \
#            sigma_2_bounds=(1e-4,1e4),\
#            gamma_bounds=(1e-5,10),gamma=5e-3) + gp.kernels.RBF()
#    gpr[i] = gp.GaussianProcessRegressor(kernel=kern[i],normalize_y=True,n_restarts_optimizer=1)
#    gpr[i].fit(ts_depths,logit_a2_neem[:,i])
#    (means_at_train[:,i],sd_at_train[:,i]) = gpr[i].predict(depths_train,return_std=True)
gpr_fab = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(300,nu=0.5) +gp.kernels.WhiteKernel(0.1))
gpr_fab.fit(depths_ts,logit(eigs_ts))


(means_at_train_logit,sd_at_train) = gpr_fab.predict(depths_train,return_std=True)
sq_precision_at_train = (1/(sd_at_train**2))[:,np.newaxis]
