import numpy as np
import scipy as sp
import tensorflow as tf
from params import *
#from sonic_data import ntrain
from fabric_gpr import means_at_train,sq_precision_at_train,means_at_train_logit
ntrain=3000
#a2 is the per-grain structure tensor in Voigt notation.
logit_c_probs = tf.Variable(init_weights((ntrain,MEPdim),1e-2),name='logitcprobs')
logit_c_probs_drop = tf.nn.dropout(logit_c_probs,keep_prob=1.0)
#change for different regularization
mult_noise = 0.0
l2_reg = 0.1

c_probs = tf.nn.softmax(logit_c_probs) #dim = (batch_size,MEPdim)
A2 = tf.matmul(c_probs,a2_meps[:,0:3]) #(batch_size,3)
A2_logit = tf_logit(A2)

#noising the logit of the softmax output instead of weights
noisy_A2_logit = A2_logit * tf.random_normal(tf.shape(A2_logit),mean=1.0,stddev=mult_noise)
noisy_obj_logit = tf.random_normal(tf.shape(A2_logit),mean=1.0,stddev=mult_noise)

ts_loss = tf.reduce_mean(sq_precision_at_train * tf.pow(A2_logit - means_at_train_logit,2)) + 0.1*tf.reduce_mean(tf.square(logit_c_probs))
regularizer = l2_reg*tf.reduce_mean(tf.square(logit_c_probs))
