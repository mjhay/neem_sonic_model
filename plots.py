#!/usr/bin/python
from params import *
import numpy as np
import scipy as sp
from fab_model import ts_loss,A2,A2_logit
#from vel_model import p_loss
#from vel_corr import fit_pwave,sqvels_modeled
#from sonic_data import sqvels_train_nd,ntrain,depths_train
from vel_model import chrisMat,char_time
import sonic_data as son
import matplotlib.pyplot as plt
from synth_sess import * 
#pwave_corr = np.zeros((ntrain,1))
#ts_optimizer = tf.train.AdagradOptimizer(1e2).minimize(ts_loss)

#def step_ts(optimizer_op,niter):
#    for i in range(0,niter):
#        sess.run(optimizer_op)
#sess.run(tf.global_variables_initializer())
#for i in range(0,10000):
#    sess.run(ts_optimizer)
#pwave_corr = fit_pwave(depths_train,sess.run(chrisMat),sqvels_train_nd)
#p_loss_with_corr = p_loss(chrisMat[:,2,2],sqvels_train_nd+pwave_corr)
#tsvel_optimizer = tf.train.AdagradOptimizer(1e0).minimize(p_loss_with_corr)
#init_uninitialized()
#for i in range(0,3000):
#    sess.run(tsvel_optimizer)
#
#def A2_samp(n):
#    A2n = np.zeros((n,ntrain,3))
#    for i in range(0,n):
#        A2n[i,:,:] = sess.run(A2)
#    A2n_mean = A2n.mean(0)
#    A2n_std = A2n.std(0)
#    return (A2n,A2n_mean,A2n_std)
#print sess.run(A2)
fig,ax = plt.subplots()
ax.scatter(depths_ts,eigs_ts[:,0],label=r'Thin section $\lambda_1$',s=20,c='b',marker='^',linewidth=0)
ax.scatter(depths_ts,eigs_ts[:,1],label=r'Thin section $\lambda_2$',s=30,c='r',marker='*',linewidth=0)
ax.scatter(depths_ts,eigs_ts[:,2],label=r'Thin section $\lambda_3$',s=20,c='k',linewidth=0)

ax.plot(depths_train,eigs_true[:,0],label=r'True $\lambda_1$',c='b',linestyle='--')
ax.plot(depths_train,eigs_true[:,1],label=r'True $\lambda_2$',c='r',linestyle='--')
ax.plot(depths_train,eigs_true[:,2],label=r'True $\lambda_3$',c='k',linestyle='--')

ax.plot(depths_train,modeled_A2[:,0],label=r'Modeled $\lambda_1$',c='b')
ax.plot(depths_train,modeled_A2[:,1],label=r'Modeled $\lambda_2$ ',c='r')
ax.plot(depths_train,modeled_A2[:,2],label=r'Modeled $\lambda_3$',c='k')
ax.set_xlim(0,3000)
ax.set_ylim(0,1)
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Eigenvalue")
ax.legend(loc='upper right',ncol=3)

figv,axv = plt.subplots()
axv.plot(depths_train,true_vel_corr[:,0],label='True velocity drift',c='k',linestyle='--')
axv.plot(depths_train,vel_corr[:,0],label=r'Est. $v_{sh}$ drift',c='b')
axv.plot(depths_train,vel_corr[:,1],label=r'Est. $v_{sh}$ drift',c='r')
axv.plot(depths_train,vel_corr[:,2],label=r'Est. $v_p$ drift',c='k')
axv.set_xlim(0,3000)
axv.set_xlabel("Depth (m)")
axv.set_ylabel("Velocity (m/s)")
fig.tight_layout()
fig.set_size_inches(12,8)
fig.savefig("eigenvalues.pdf",format='pdf')

axv.legend(loc='upper left')
figv.tight_layout()
figv.set_size_inches(12,8)
figv.savefig("vel_correction.pdf",format='pdf')
