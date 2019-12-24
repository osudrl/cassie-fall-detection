import numpy as np
import gym
import torch
import argparse
import pickle

state_vector = []
state_labels = []

for i in range(30):
    v = "state_vector" + str(i + 1) + ".p"
    l = "state_labels" + str(i + 1) + ".p"
    v = (pickle.load(open("./data/" + v, "rb" ))).tolist()
    l = (pickle.load(open("./data/" + l, "rb" ))).tolist()
    state_vector.append(v)
    state_labels.append(l)

state_vector = np.reshape(np.ndarray.flatten(np.array(state_vector)), (-1, 46))
state_labels = np.ndarray.flatten(np.array(state_labels))

print(state_vector)
print(state_labels)

pickle.dump(state_vector, open("./data/" + "state_vector.p", "wb"))
pickle.dump(state_labels, open("./data/" + "state_labels.p", "wb"))
