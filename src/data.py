import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


A = np.array([[0.6, 0.3, 0.1], [0.1,0.8,0.1],[0.1,0.3,0.6]])
print(A)

mu = np.array([-2, 0, 2])
print(mu)

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
        print(update)
init = np.array([0.2,0.6,0.2])

random.seed(1)
state = np.zeros(1000)
obs = np.zeros(1000)
state[0] = generate_state(init, generate_num())
obs[0] = generate_obs(state[0])
print(int(state[0]))

for i in np.arange(1, 1000):
    tran = A[int(state[i-1])]
    state[i] = generate_state(tran, generate_num())
    obs[i] = generate_obs(state[i])

plt.plot(obs)
plt.show()
plt.plot(state)
plt.show()