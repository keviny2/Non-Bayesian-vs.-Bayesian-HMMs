import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import BayesianHMM
import MaxLikeHMM

from Distribution import normal_pdf

A = np.array([[0.6, 0.3, 0.1], [0.1,0.8,0.1],[0.1,0.3,0.6]])
mu = np.array([-2, 0, 2])


def marginal(A, init):
    return np.dot(A.T, init)

def generate_num():
    return random.uniform(0, 1)

def generate_state(v, num):
    n = len(v)
    for i in range(3):
        if num < sum(v[:i+1]):
            a = i
            return a

def generate_obs(state):
    if state == 0:
        a = np.random.normal(-2, 1)
    elif state == 1:
        a = np.random.normal(0, 1)
    else:
        a = np.random.normal(2, 1)
    return a

converge = False
init = np.array([1/3,1/3,1/3])
while not converge:
    update = marginal(A, init)
    if ((update != init).all()):
        init = update
    else:
        converge = True
init = np.array([0.2,0.6,0.2])

state = np.zeros(1000)
obs = np.zeros(1000)
state[0] = generate_state(init, generate_num())
obs[0] = generate_obs(state[0])

for i in np.arange(1, 1000):
    tran = A[int(state[i-1])]
    state[i] = generate_state(tran, generate_num())
    obs[i] = generate_obs(state[i])

plt.figure()
plt.plot(obs)
fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "original")
plt.savefig(fname)
print("\nFigure saved as '%s'" % fname)
# plt.plot(state)
# plt.show()

model = MaxLikeHMM.MaxLikeHMM(obs)
A = np.array([[0.6, 0.2,0.2], [0.2,0.6,0.2], [0.2,0.2,0.6]])
B = np.array([[-2, 3], [0, 3], [2,3]])
init = np.array([0.3, 0.4,0.3])
sim_A, sim_B, sim_init = model.baum_welch_robust(A, B, init)

path, _, _ = model.viterbi_robust(sim_init, sim_A, sim_B)
print(path)
print(sum(path == state))

sim_obs = np.zeros(1000)
obs[0] = np.random.normal(0, 1)
for i in np.arange(1, 1000):
    tran = sim_A[int(state[i-1])]
    state[i] = generate_state(tran, generate_num())
    sim_obs[i] = np.random.normal(0, 1)

plt.figure()
plt.plot(sim_obs)
fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "simulate")
plt.savefig(fname)
print("\nFigure saved as '%s'" % fname)
