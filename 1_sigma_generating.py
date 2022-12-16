
# Deviations Approximation


import copy
import cv2
import matplotlib as plt
import numpy as np
import pandas as pd

img = cv2.imread('observation2sample_02.png')
rows = img.shape[0]
cols = img.shape[1]

img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_Gray, type(img_Gray))
# print(img_Gray.shape[0], img_Gray.shape[1])

img_cover = copy.deepcopy(img_Gray)
for i in range(rows):
    for j in range(cols):
        img_cover[i][j] = 255 - img_cover[i][j]
# print(img_cover, img_cover.shape[0], img_cover.shape[1])

l1 = []
l2 = []
observations = []
for i in range(rows):
    for j in range(cols):
        l1.append([i, j])

data = img_cover.flatten()

k = 0
for i in range(rows * cols):
    for j in range(data[k]):
        observations.append(l1[i])
    k += 1
observations = np.array(observations)
#此处需要对observations作图！！--------2022.12.13已完成。


data_row = []
data_col = []
for i in range(len(observations)):
    data_row.append(observations[i][0])
    data_col.append(observations[i][1])

data_row_ary = np.array(data_row)
data_col_ary = np.array(data_col)

a = np.mean(data_row_ary)
b = np.mean(data_col_ary)

mu_obs = np.array([a, b])

# mu0 = np.array([0, 0])
# cov = np.array([[1, 0.1], [0.1, 1]])
# Proposal = lambda X: np.random.multivariate_normal([X[0], X[1]], cov=cov)
Proposal = lambda X: [np.random.normal(X[0], 0.3, (1,)), np.random.normal(X[1], 0.3, (1,))]

def prior(X):
    if X[0] > 0 and X[1] > 0:
        return 1
    return 0

# def rho_clac(X):
#     return ((np.sum(data_row_ary * data_col_ary) / np.size(data_row_ary) * np.size(data_col_ary)) -
#             mu_obs[0] * mu_obs[1]) / X[0] * X[1]


def manual_log_like_normal(X):
    return np.sum(-np.log(2 * np.pi * X[0] * X[1]) -
                  (1 / 2) * (((data_row_ary - mu_obs[0]) / X[0]) ** 2 + ((data_col_ary - mu_obs[1]) / X[1]) ** 2))

def acceptance(X, X_new):
    if X_new > X:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (np.exp(X_new - X)))

def metropolis_hastings(lik_clac, pri, proposal, parameter_init, iterations, obs, acceptance_rule):
    X = parameter_init
    accepted = []
    for i in range(iterations):
        X_new = proposal(X)
        X_lik = lik_clac(X)
        X_new_lik = lik_clac(X_new)
        if (acceptance_rule(X_lik + np.log(pri(X)), X_new_lik + np.log(pri(X_new)))):
            X = X_new
            accepted.append(X_new)
    return np.array(accepted)


accepted = metropolis_hastings(manual_log_like_normal, prior, Proposal, [0.1, 0.1], 10000, observations, acceptance)
# accepted_indeed = accepted[0][500:]

accepted = accepted.reshape(-1, 2)
# print(accepted)
# np.savetxt('sigma_generated_03.txt', accepted, fmt='%.5f', delimiter=',')
data = pd.DataFrame(accepted)
data.to_csv('sigma_03.csv', index=None)


print(len(accepted), accepted.shape, type(accepted))
# with open('sig_gen.txt', 'w') as file_01:
#     for slice_2d in accepted:
#         np.savetxt(file_01, slice_2d, fmt='%.3f', delimiter=',')

# print(accepted_indeed)