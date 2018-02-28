#!/usr/bin/python
from params import *
import numpy as np
import scipy as sp
from fab_model import ts_loss,A2,A2_logit,regularizer
from fabric_gpr import depths_ts
from vel_loss import p_loss
from vel_corr import sqvels_modeled,fit_vels
from sonic_data import sqvels_train_nd,vels_train
from vel_model import chrisMat,sqvels, sqvels_chris, vel_loss
from synth_eigs import softmax
from fabric_gpr import means_at_train_logit
from chris import char_time
import matplotlib.pyplot as plt
def to_sqvels_nd(vels):
    return vels**2*char_time**2

def to_dimvels(sqvels):
    return np.sqrt(sqvels)/char_time

depths_train = np.linspace(0,3000,3000)[:,np.newaxis]
depths_ts = depths_train[::30,:]
ntrain = 3000
#pwave_corr = np.zeros((ntrain,1))
ts_optimizer = tf.train.AdagradOptimizer(1e2).minimize(ts_loss)

def step_ts(optimizer_op,niter):
    for i in range(0,niter):
        sess.run(optimizer_op)

sess.run(tf.global_variables_initializer())

for i in range(0,1000):
    sess.run(ts_optimizer)

vel_corr_gpr = fit_vels(depths_train[::30,:],to_dimvels(sess.run(sqvels_chris)[::30,:]),to_dimvels(sqvels_train_nd[::30,:]))
vel_corr = vel_corr_gpr.predict(depths_train)
#vel_corr = np.sqrt(veld_corr)/char_time
corrected_sqvels = to_sqvels_nd(vels_train+vel_corr)
vel_loss_with_corr = vel_loss(sqvels,corrected_sqvels)
p_loss_with_corr = vel_loss(sqvels,corrected_sqvels)
#p_loss_with_corr = p_loss(sess.run(chrisMat[:,2,2]),sqvels_train_nd)
tsp_optimizer = tf.train.AdagradOptimizer(1e1).minimize(p_loss_with_corr+ts_loss+1e1*regularizer)
tsvel_optimizer = tf.train.AdagradOptimizer(1e-4).minimize(vel_loss_with_corr+ts_loss+1e1*regularizer)
init_uninitialized()
for i in range(0,2000):
    sess.run(tsp_optimizer)
#
#
for i in range(0,5000):
    sess.run(tsvel_optimizer)
#
##def A2_samp(n):
##    A2n = np.zeros((n,ntrain,3))
##    for i in range(0,n):
##        A2n[i,:,:] = sess.run(A2)
##    A2n_mean = A2n.mean(0)
##    A2n_std = A2n.std(0)
##    return (A2n,A2n_mean,A2n_std)
eigs_ts = np.genfromtxt("synth_eigs_ts.csv",delimiter=' ',skip_header=True)
eigs_true = np.genfromtxt("synth_eigs_true.csv",delimiter=' ',skip_header=True)
vels_true = np.genfromtxt("vels.csv",delimiter=' ',skip_header=True)
vels_corrup = np.genfromtxt("vels_with_corruption.csv",delimiter=' ',skip_header=True)
vels_true_nd = to_sqvels_nd(vels_true)
mean_ts = softmax(means_at_train_logit)
vels_corrup_nd = to_sqvels_nd(vels_corrup)
modeled_A2=sess.run(A2)
#
true_vel_corr = vels_true - vels_corrup
print sess.run(A2)
plt.scatter(depths_ts.repeat(3,1),eigs_ts);plt.plot(eigs_true,c='k');plt.plot(sess.run(A2));plt.plot(mean_ts);plt.show()
plt.plot(vels_true-vels_corrup);plt.plot(vel_corr)
#plt.plot(vel_corr);plt.plot(vels_true_nd-vels_corrup_nd)
