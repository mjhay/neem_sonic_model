from params import *
from synth_eigs import depths_train,depths_ts,logit
from sklearn.cluster import KMeans
import sklearn.gaussian_process as gp
from gp_extras.kernels import HeteroscedasticKernel
#fit smooth eigenvalues to thin sections
prototypes = KMeans(n_clusters=10).fit(ts_depths).cluster_centers_

kern=[0,0,0]
gpr=[0,0,0]
means_at_train = np.zeros((depths_train.shape[0],3))
eigs_ts = np.genfromtxt(fabric_file,delimiter=' ',skip_header=1)
sd_at_train = 0.1*np.ones((depths_train.shape[0],3))
gpr_fab = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(300,nu=0.5) +gp.kernels.WhiteKernel(0.1))
gpr_fab.fit(depths_ts,logit(eigs_ts))


(means_at_train_logit,sd_at_train) = gpr_fab.predict(depths_train,return_std=True)
sq_precision_at_train = (1/(sd_at_train**2))[:,np.newaxis]
