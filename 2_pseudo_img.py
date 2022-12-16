
# Simulation Image Implementation

import numpy as np
import pandas as pd
from pandas import read_csv
import cv2
import copy
import PIL.Image as Image


filename = 'sigma_03.csv'
data = read_csv(filename)
data_01 = data.values
# print(data_01[-1])


img = cv2.imread('observation2sample_02.png')
ROWS = img.shape[0]
COLS = img.shape[1]
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
num2gen = np.sum(img_Gray.flatten())
num2gen_new = int(np.around(num2gen * 0.1))
# print(num2gen)

# img_cover = copy.deepcopy(img_Gray)
# for i in range(ROWS):
#     for j in range(COLS):
#         img_cover[i][j] = 255 - img_cover[i][j]

# print(ROWS, COLS)
mu_01 = np.random.uniform(0, ROWS)
mu_02 = np.random.uniform(0, COLS)
# print(mu_02)
# mean = np.array([mu_01, mu_02])
# cov = np.array([[data_01[-1][0], 0], [0, data_01[-1][1]]])



pic_01 = []
pic_02 = []
# itertions = 1000
# for i in range(itertions):
for j in range(num2gen_new):
    # X = np.random.multivariate_normal(mean, cov, size=None)
    X_1 = np.random.normal(mu_01, data_01[-1][0])
    X_2 = np.random.normal(mu_02, data_01[-1][1])
    X_new_1 = np.around(X_1)
    X_new_2 = np.around(X_2)
    pic_01.append(X_new_1)
    pic_02.append(X_new_2)
pic_03 = np.array(pic_01)
pic_04 = np.array(pic_02)
pic_05 = pic_03.astype(int)
pic_06 = pic_04.astype(int)
# print(pic_02, pic_02.shape)
# print(pic_02.shape)
# pic_07 = sorted(pic_02, key=lambda i: i[0])
# print(pic_07)
# pic_08 = sorted(pic_07, key=lambda i: i[1])
# print(pic_08)
# pic_08 = np.array(pic_08)

# l0 = []
# for i in range(len(pic_08)):
#     y = 0
#     for j in range(len(pic_08)):
#         if pic_08[i][0] == pic_08[j][0] and pic_08[i][1] == pic_08[j][1]:
#             y += 1
#         # y -= 1
#     l0.append(y)
# print(l0)





# l1 = l1.tolist()
# print(type(l1), type(l1[0]))
# for i in l1:
#     i.tolist()
# print(type(l1[0]))
# l2 = np.unique(l1)
# print(l1, l2)
# l2 = list(set(l1))
# l3 = []
# for i in l2:
#     l3.append(l1.count(i))
# print(l2, l3)
# print(l2)

pic_07 = []
for i in range(ROWS):
    for j in range(COLS):
        y = 0
        # z = np.array([i, j])
        for k in range(num2gen_new):
            if i == pic_05[k] and j == pic_06[k]:
                y += 1
        pic_07.append(y)
#
pic_08 = np.array(pic_07)
pic_09 = pic_08.reshape(ROWS, -1)
# pic_011 = []
for i in range(pic_09.shape[0]):
    for j in range(pic_09.shape[1]):
        pic_09[i][j] = pic_09[i][j] * 3
for i in range(pic_09.shape[0]):
    for j in range(pic_09.shape[1]):
        if pic_09[i][j] <= 10:
            pic_09[i][j] = 0
for i in range(1, pic_09.shape[0] - 1, 2):
    for j in range(1, pic_09.shape[1] - 1, 2):
        if pic_09[i-1][j] == 0 and pic_09[i+1][j] == 0 and pic_09[i][j-1] == 0 and pic_09[i][j+1] == 0:
            pic_09[i][j] = 0



        # pic_011.append(a)
# pic_09 = np.array(pic_011)
# pic_09 = pic_09.reshape(ROWS, -1)

# for i in range(pic_09.shape[0]):
#     for j in range(pic_09.shape[1]):
#         pic_09[i][j] = 255 - pic_09[i][j]
# pic_05 = pic_04.reshape(ROWS, COLS)
print(pic_09, pic_09.shape)

pic_010 = Image.fromarray(pic_09)
pic_010 = pic_010.convert('RGB')
pic_010.save('010.png')