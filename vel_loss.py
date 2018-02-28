from vel_model import *
from sonic_data import sqvels_train_nd
p_loss = tf.reduce_mean(pvel_sd_nd**-2*tf.square(sqvels_chris[:,2]-sqvels_train_nd[:,2]))    
#p_loss = tf.reduce_mean(pvel_sd_nd**-2*tf.square(chrisMat[:,2,2]-sqvels_train_nd[:,0]))    
vel_loss = tf.reduce_mean(vel_precisions**2 * tf.square(sqvels_chris-sqvels_train_nd))
