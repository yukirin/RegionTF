# sample.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================
import tensorflow as tf

a = tf.constant("hello tf")
sess = tf.Session()
print(sess.run(a))
