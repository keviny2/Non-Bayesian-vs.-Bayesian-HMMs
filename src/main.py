import numpy as np
import pandas as pd
import function

obs_map = {'Cold' :0, 'Hot' : 1}
obs = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1])
inv_obs_map = dict([(value,key) for key, value in obs_map.items()])
hidden_map = {'Snow': 0, 'Rain': 1, 'Sunshine': 2}
inv_hidden_map = dict((value, key) for key, value in hidden_map.items())
# inv_obs_map = {}
# for key, value in obs_map.items():
#    if value in new_dict:
#        inv_obs_map[value].append(key)
#    else:
#        inv_obs_map[value]= [key]
#
# print(inv_obs_map)

obs_seq = np.array([inv_obs_map[v] for v in obs])

# obs_seq = np.array([])
# for v in obs:
#     obs_seq = obs_seq = np.append(obs_seq, inv_obs_map[v])
# print(obs_seq)

sim_obs = pd.DataFrame(np.transpose(np.stack((obs, obs_seq))), columns=['code', 'obs'])
init = np.array([0.6, 0.4])
obs_states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']

pi = [0, 0.2, 0.8]
hidden_clusters = pd.Series(pi, index = hidden_states, name = 'state')

transition_matrix = np.array([[0.3, 0.3,0.4], [0.1,0.45,0.45], [0.2,0.3,0.5]])
transition = pd.DataFrame(transition_matrix, columns = hidden_states, index = hidden_states)

emission_matrix = np.array([[1, 0], [0.8,0.2], [0.3,0.7]])
emission = pd.DataFrame(emission_matrix, columns = obs_states,index = hidden_states)

path, prob, state = function.viterbi(pi, transition_matrix, emission_matrix,obs )
print(path, prob, state)