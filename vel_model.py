import numpy as np
import scipy as sp
import tensorflow as tf
from params import *
from fab_model import c_probs
from chris import chris_meps, char_time
from sonic_data import sqvels_train_nd
#just SQUARED pvels, no shears
#nondimensionalize by char_time first
def pvel_loss(sqvels,obs_vels,p_sigma):
    return pvel_sd_nd**-2*tf.reduce_mean(tf.square(sqvels[:,0]-obs_vels[:,0]))    

pvel_sd = 10. #m/s
sh_sd = 40.
pvel_sd_nd = pvel_sd*char_time
sh_sd_nd = sh_sd*char_time
vel_precisions = np.array([1./sh_sd_nd,1./sh_sd_nd,1./pvel_sd_nd])
chrisMat = tf.tensordot(c_probs,chris_meps,[[1],[0]])


psh2 = tf.expand_dims(chrisMat[:,0,0],-1)
psv2 = tf.expand_dims(chrisMat[:,1,1],-1)
pp2 = tf.expand_dims(chrisMat[:,2,2],-1)
sqvels = tf.concat([psh2,psv2,pp2],-1)
sqvels_chris = sqvels
def p_loss(sq_pvels,sqvels_train):
    return tf.reduce_mean(vel_precisions[2]**(2)*tf.square(sq_pvels-sqvels_train[:,0]))    

def vel_loss(vels,vels_train):
    return tf.reduce_mean(tf.square((vels-vels_train))*vel_precisions**2)
