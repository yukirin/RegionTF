# mpmodel.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================

"""Example / benchmark for building a LSTM model.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

To run:

$ python mpmodel.py

"""

import time
import numpy as np
import tensorflow as tf
import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class MPInput(object):
  """The input data."""

  def __init__(self, config, data=None, name=None):
    self.input_size = config.input_size
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.output_size = config.output_size
    if data is None:
      return
    
    self.epoch_size = (len(data) - num_steps - 1) // batch_size
    self.input_data, self.targets = reader.data_producer(
        data, batch_size, num_steps, config.output_size, name=name)


class MPModel(object):
  """The MP model."""

  def __init__(self, is_training, config, input_, is_test=False):
    self._input = input_

    input_size = config.input_size
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    output_size = config.output_size

    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)

    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      if is_test:
        p_input = tf.placeholder(tf.float32, [num_steps, input_size])
        inputs = tf.reshape(p_input, shape=[1, num_steps, input_size])
      else:
        inputs = input_.input_data

    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs, initial_state=self._initial_state)

    output = outputs[-1]
    softmax_w = tf.get_variable(
        "softmax_w", [size, output_size * input_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [output_size * input_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b

    targets = tf.reshape(input_.targets, [-1, output_size * input_size])
    loss = tf.losses.mean_squared_error(targets, logits)
    self._logits = logits
    self._cost = loss
    self._final_state = state

    if not is_training:
      return

    global_step = tf.contrib.framework.get_or_create_global_step()
    self._lr = tf.train.exponential_decay(config.learning_rate, global_step,
                                          input_.epoch_size * config.max_epoch,
                                          config.lr_decay, staircase=True)

    adam = tf.train.AdamOptimizer(learning_rate=self._lr)
    gradients = adam.compute_gradients(loss)
    grads, tvars = zip(*gradients)
    grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
    self._train_op = adam.apply_gradients(zip(grads, tvars), global_step=global_step)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def logits(self):
    return self._logits


class SmallConfig(object):
  """Small config."""
  input_size = 3
  init_scale = 0.04
  learning_rate = 0.01
  max_grad_norm = 5
  num_layers = 3
  num_steps = 40
  hidden_size = 200
  max_epoch = 5
  max_max_epoch = 100
  keep_prob = 1
  lr_decay = 0.97
  batch_size = 30
  output_size = 30


def run_epoch(session, model, eval_op=None, verbose=False, is_test=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "predict": model.logits,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    costs += cost

    if verbose and step % 100 == 0 and step > 0:
      print("%.8f cost average: %.8f speed: %.1f bps" %
            (step * 1.0 / model.input.epoch_size, costs / (step + 1),
             step / (time.time() - start_time)))

  return costs / model.input.epoch_size


def predict_epoch(session, model, offset=0, verbose=False):
  """Runs the model on the given data."""
  raw = reader.pos_raw_data()
  data = np.array(raw).reshape((len(raw), model._input.input_size))
  input_data = data[offset:offset + model.input.num_steps]

  fetches = {
      "predict": model.logits,
  }

  feed_dict = {"Test/Model/Placeholder:0": input_data}
  vals = session.run(fetches, feed_dict)

  actual = raw[offset:offset + model.input.num_steps + model.input.output_size]
  p = np.reshape(vals["predict"], [model.input.output_size, model.input.input_size])
  predict = np.r_[input_data, p]
  return actual, predict


def get_config():
  return SmallConfig()


def main(_):
  raw_data = reader.pos_raw_data()
  train_data = test_data = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = MPInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = MPModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Test"):
      test_input = MPInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = MPModel(is_training=False, config=eval_config,
                        input_=test_input, is_test=True)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=180)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

        train_cost_avg = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        print("Epoch: %d Train Cost Avg: %.8f" % (i + 1, train_cost_avg))

      actual, predicts = predict_epoch(session, mtest, offset=50)


if __name__ == "__main__":
  tf.app.run(main=main)