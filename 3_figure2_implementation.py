
# Histogram Image Implementation

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('observation2sample_02.png')
rows = img.shape[0]
cols = img.shape[1]

img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_cover = copy.deepcopy(img_Gray)
for i in range(rows):
    for j in range(cols):
        img_cover[i][j] = 255 - img_cover[i][j]


l1 = []
l2 = []
observations = []
for i in range(rows):
    for j in range(cols):
        l1.append([i, j])

data = img_cover.flatten()

# k = 0
# for i in range(rows * cols):
#     for j in range(data[k]):
#         observations.append(l1[i])
#     k += 1
# observations = np.array(observations)

# print(observations.shape)

#提取数组中的每个元素：
X = []
Y = []
for i in range(rows):
    X.append(i)
for i in range(cols):
    Y.append(i)
X = np.array(X)
Y = np.array(Y)

# print(X.shape == Y.shape)
# print(len(data), len(X), len(Y))
# print(rows, cols)
# print(X, Y)

XX, YY = np.meshgrid(X, Y)
X = XX.ravel()
Y = YY.ravel()
height = np.zeros_like(data)
width = depth = 0.3
c = ['r'] * len(data)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.bar3d(X, Y, height, width, depth, data, color=c, shade=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('figure2.png')
plt.show()