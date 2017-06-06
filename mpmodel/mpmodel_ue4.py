# mpmodel_ue4.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================
import tensorflow as tf
import numpy as np

import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

import mpmodel

class MPModelUE4(TFPluginAPI):
   def setup(self):
       eval_config = mpmodel.get_config()
       eval_config.batch_size = 1
       self.num_steps = eval_config.num_steps
       self.output_size = eval_config.output_size
       self.input_size = eval_config.input_size

       with tf.Graph().as_default():
           initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
           with tf.name_scope("Test"):
              test_input = mpmodel.MPInput(config=eval_config, data=test_data,
                                           is_predict=True, name="TestInput")
              with tf.variable_scope("Model", reuse=False, initializer=initializer):
                 self.model = mpmodel.MPModel(is_training=False, config=eval_config,
                                                               input_=test_input, is_test=True)

       self.sess = tf.InteractiveSession()
       ckpt = tf.train.get_checkpoint_state('./save')
       last_model = ckpt.model_checkpoint_path
       saver = tf.train.Saver()
       saver.restore(sess, last_model)

   def _decodeJson(self, jsonInput):
      data = np.array(jsonInput['input'].split(','), dtype=np.float32)
      data = np.reshape(data, [self.num_steps, self.input_size])
      return data

   def _encodeJson(predict_data):
      result = {}
      for i, pos in enumerate(predict_data):
        result[str(i)] = ",".join(str(n) for n in pos)
      return result

   def runJsonInput(self, jsonInput):
       input_data = self._decodeJson(jsonInput)
       feed_dict = {"Test/Model/Placeholder:0": input_data}
       return self._encodeJson(self.sess.run(self.model.logits, feed_dict)) 

   def train(self):
       pass

def getApi():
    return MPModelUE4.getInstance()