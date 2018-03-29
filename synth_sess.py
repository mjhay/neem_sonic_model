#!/usr/bin/python
from params import *
import numpy as np
import scipy as sp
from fab_model import ts_loss,A2,A2_logit,regularizer
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

#create ts_interval00 synthetic sonic velocity depths
ntrain = 3000
#thin section every ts_intervalm.
ts_interval = 30
depths_train = np.linspace(0,ntrain,ntrain)[:,np.newaxis]
depths_ts = depths_train[::ts_interval,:]
#optimizer to fit velocities to synthetic thin sections.
ts_optimizer = tf.train.AdagradOptimizer(1e2).minimize(ts_loss)

#run it
sess.run(tf.global_variables_initializer())

for i in range(0,1000):
    sess.run(ts_optimizer)

#velocity prediction
vel_corr_gpr = fit_vels(depths_train[::ts_interval,:],to_dimvels(sess.run(sqvels_chris)[::ts_interval,:]),to_dimvels(sqvels_train_nd[::ts_interval,:]))
vel_corr = vel_corr_gpr.predict(depths_train)
#correct velocities
corrected_sqvels = to_sqvels_nd(vels_train+vel_corr)
vel_loss_with_corr = vel_loss(sqvels,corrected_sqvels)
p_loss_with_corr = vel_loss(sqvels,corrected_sqvels)
tsp_optimizer = tf.train.AdagradOptimizer(1e1).minimize(p_loss_with_corr+ts_loss+1e1*regularizer)
tsvel_optimizer = tf.train.AdagradOptimizer(1e-4).minimize(vel_loss_with_corr+ts_loss+1e1*regularizer)
init_uninitialized()
for i in range(0,2000):
    sess.run(tsp_optimizer)
for i in range(0,5000):
    sess.run(tsvel_optimizer)

eigs_ts = np.genfromtxt("synth_eigs_ts.csv",delimiter=' ',skip_header=True)
eigs_true = np.genfromtxt("synth_eigs_true.csv",delimiter=' ',skip_header=True)
vels_true = np.genfromtxt("vels.csv",delimiter=' ',skip_header=True)
vels_corrup = np.genfromtxt("vels_with_corruption.csv",delimiter=' ',skip_header=True)
vels_true_nd = to_sqvels_nd(vels_true)
mean_ts = softmax(means_at_train_logit)
vels_corrup_nd = to_sqvels_nd(vels_corrup)
modeled_A2=sess.run(A2)
true_vel_corr = vels_true - vels_corrup
print sess.run(A2)

plt.scatter(depths_ts.repeat(3,1),eigs_ts);plt.plot(eigs_true,c='k');plt.plot(sess.run(A2));plt.plot(mean_ts);plt.show()
plt.plot(vels_true-vels_corrup);plt.plot(vel_corr)
