#!/usr/bin/python
from sess import *
import matplotlib.pyplot as plt
from vel_model import *
from chris import *
from fab_data import *
for i in xrange(0,100):
    step_ts()
preds = sess.run(A2,feed_dict={X_ts:ts_depths_tr})
plt.plot(preds)
plt.plot(a2_neem)
plt.show()
predvels = sess.run(sqvels_chris,feed_dict={X_ts:depths_norm})
plt.plot(predvels);plt.show()
plt.plot(sqvels);plt.show()
