# reader.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================

import math
import csv
import numpy as np
import tensorflow as tf


def sin_raw_data():
  steps_par_cycle = 50
  number_of_cycles = 100
  x = np.sin(np.arange(0, number_of_cycles * steps_par_cycle + 1, 2 * math.pi / steps_par_cycle))
  x = (x + 1) / 2
  return x


def pos_raw_data():
  with open("data.csv") as f:
    r = csv.reader(f)
    d = np.array(list(r), dtype=np.float32)
    return d


def data_producer(raw, batch_size, num_steps, predict_size=1, name=None):
    data = tf.convert_to_tensor(raw, name="raw_data", dtype=tf.float32)

    if np.linalg.matrix_rank(raw) == 1:
      input_size = 1
    else:
      input_size = raw.shape[1]

    epoch_size = (len(raw) - num_steps - predict_size) // batch_size
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    data_list = []
    predicts = []
    for c in range(batch_size):
        pos = batch_size * i + c
        data_list.append(tf.reshape(data[pos: pos + num_steps], shape=[1, num_steps, input_size]))
        predict = data[pos + num_steps: pos + num_steps + predict_size]
        predicts.append(tf.reshape(predict, shape=[1, predict_size, input_size]))

    x = tf.concat(data_list, 0)
    y = tf.concat(predicts, 0)
    return x, y


def main(_):
    x, y = data_producer(pos_raw_data(), 2, 3, 2, name="DataProducer")

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        print(sess.run([x, y]))


if __name__ == '__main__':
    tf.app.run()
