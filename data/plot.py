# plot.py
#
# Copyright (c) [2017] [yukirin]
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ==============================================================================

import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

with open("posdata.csv", 'r') as f:
    r = csv.reader(f)
    d = np.array(list(r), dtype=np.float32)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(x, y, z)
plt.show()
