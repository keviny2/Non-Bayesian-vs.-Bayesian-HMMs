import numpy as np
import pandas as pd

def viterbi(pi, transition, emission, obs):
    hidden = np.shape(emission)[0]
    d = np.shape(obs)[0]

    # init blank path
    path = np.zeros(d, dtype = int)
    #  highest probability of any path that reaches state i
    prob = np.zeros((hidden, d))
    # the state with the highest probability
    state = np.zeros((hidden, d))

     # init delta and phi
    prob[:, 0] = pi * emission[:, obs[0]]
    state[:, 0] = 0

    for i in range(1, d, 1):
        for j in range(hidden):

            prob[j,i] = np.max(prob[:, i-1] * transition[:, j] * emission[j, obs[i]])
            state[j,i] = np.argmax(prob[:, i-1] * transition[:, j] * emission[j, obs[i]])

    path[d-1] = np.argmax(prob[:, d-1])
    for i in range(d-2, -1, -1):
        path[i] = state[path[i+1], [i+1]]

    return path, prob, state