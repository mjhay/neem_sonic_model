from params import *
from vel_model import chrisMat, sqvels_chris
from sonic_data import sqvels_train_nd
from chris import char_time
from sonic_data import sqvels_train_nd,vels_train
import sklearn.kernel_ridge as kr
import sklearn.gaussian_process as gp
init_uninitialized()
sd_corr = 10#char_time
kern_pwave = gp.kernels.WhiteKernel(noise_level=sd_corr) + gp.kernels.RBF(length_scale=5e2)
sqvels_modeled = sess.run(sqvels_chris)
vels_modeled = np.sqrt(sqvels_modeled)/char_time
gpr_pwave = gp.GaussianProcessRegressor(kernel=kern_pwave)

def fit_vels(depths,vels_modeled,vels_train):
    y = vels_modeled - vels_train
    y = y
    print y.shape
    print depths.shape
    regr=kr.KernelRidge(alpha=0.1,kernel=gp.kernels.RBF(1000))
    regr.fit(depths,y)
#    gpr_pwave.fit(depths,y)
    return regr#gpr_pwave
