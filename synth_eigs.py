import sklearn.gaussian_process as gp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process.kernels as k

def softmax2(x):
    return np.exp(x)/np.sum(np.exp(x),-1)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),1)[:,np.newaxis]

def logit(x):
    logx = np.log(x)
    return logx - logx[:,-1][:,np.newaxis]

depths_train = np.linspace(0,3000,3000)[:,np.newaxis]
depths_ts = depths_train[::30,:]

matk_fabric = k.Matern(length_scale=300.0,nu=0.5)
kern_p = k.WhiteKernel(noise_level=5.0) + matk_fabric
kern_sh = k.WhiteKernel(noise_level=5.0) + matk_fabric
kern_sv = k.WhiteKernel(noise_level=5.0) + matk_fabric

matk_vel_error = 100*k.RBF(length_scale=600)
kern_a11 = k.WhiteKernel(noise_level=0.0001) + matk_fabric
kern_aii_noise = k.WhiteKernel(noise_level=0.2) + matk_fabric
kern_a22 = k.WhiteKernel(noise_level=10.0) + matk_fabric
kern_a33 = k.WhiteKernel(noise_level=1.0) + matk_fabric
kern_a22 = k.WhiteKernel(noise_level=10.0) + matk_fabric
kern_a33 = k.WhiteKernel(noise_level=1.0) + matk_fabric



gpr = gp.GaussianProcessRegressor(matk_fabric)
gpr_noise = gp.GaussianProcessRegressor(k.WhiteKernel(0.05))
aii = gpr.sample_y(depths_train,3)
aii += np.array([-1,0,2])
aii.sort(1)
aii_ts = aii[::30,:]
aii_noise = aii_ts + gpr_noise.sample_y(depths_ts,3)
aii_noise.sort(1)
gpr_vel = gp.GaussianProcessRegressor(kernel=matk_vel_error+k.WhiteKernel(0.001))
vel_corruption =gpr_vel.sample_y(depths_train) 

