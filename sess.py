#!/usr/bin/python
from params import *
import numpy as np
import scipy as sp
from fab_model import ts_loss,A2,A2_logit
from vel_model import p_loss
from vel_corr import fit_pwave,sqvels_modeled
from sonic_data import sqvels_train_nd,ntrain,depths_train
from vel_model import chrisMat
import matplotlib.pyplot as plt
pwave_corr = np.zeros((ntrain,1))
ts_optimizer = tf.train.AdagradOptimizer(1e2).minimize(ts_loss)

def step_ts(optimizer_op,niter):
    for i in range(0,niter):
        sess.run(optimizer_op)
sess.run(tf.global_variables_initializer())
for i in range(0,1000):
    sess.run(ts_optimizer)
pwave_corr = fit_pwave(depths_train,sess.run(chrisMat),sqvels_train_nd)
p_loss_with_corr = p_loss(chrisMat[:,2,2],sqvels_train_nd+pwave_corr)
tsvel_optimizer = tf.train.AdagradOptimizer(1e0).minimize(p_loss_with_corr)
init_uninitialized()
for i in range(0,3000):
    sess.run(tsvel_optimizer)

def A2_samp(n):
    A2n = np.zeros((n,ntrain,3))
    for i in range(0,n):
        A2n[i,:,:] = sess.run(A2)
    A2n_mean = A2n.mean(0)
    A2n_std = A2n.std(0)
    return (A2n,A2n_mean,A2n_std)
print sess.run(A2)

