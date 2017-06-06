# mpmodel_ue4.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================

import tensorflow as tf
import numpy as np

import mpmodel
import reader
import plot


def inference(_):
  test_data = reader.pos_raw_data()

  config = mpmodel.get_config()
  eval_config = mpmodel.get_config()
  eval_config.batch_size = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Test"):
      test_input = mpmodel.MPInput(config=eval_config, data=test_data, name="TestInput")

      with tf.variable_scope("Model", reuse=False, initializer=initializer):
        mtest = mpmodel.MPModel(is_training=False, config=eval_config,
                                input_=test_input, is_test=True)

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(mpmodel.FLAGS.save_path)
      if ckpt is None:
        return

      last_model = ckpt.model_checkpoint_path
      print("load " + last_model)
      saver = tf.train.Saver()
      saver.restore(sess, last_model)

      actual, predicts = mpmodel.predict_epoch(sess, mtest, offset=np.random.randint(0, 200))
      plot.plot3d(actual, predicts, config.num_steps)


if __name__ == '__main__':
  tf.app.run(main=inference)
