# plot.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot(actual, predict):
  actual = np.squeeze(actual)
  predict = np.squeeze(predict)
  x = range(0, len(actual))

  plt.title("NN [LSTM] $\sin(x)$")
  plt.plot(x, actual, label="Input")
  plt.plot(x, predict, label="Predict", linestyle="dashed")
  plt.legend()
  plt.show()


def plot3d(actual, predict, num_steps):
  d = np.array(predict, dtype=np.float32)
  x, y, z = d[:, 0], d[:, 1], d[:, 2]

  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter3D(x[:num_steps], y[:num_steps], z[:num_steps], c='blue')
  ax.scatter3D(x[num_steps:], y[num_steps:], z[num_steps:], c='red')

  plt.xlim(-1, 1)
  plt.show()
