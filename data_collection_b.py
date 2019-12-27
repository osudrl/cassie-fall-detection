import numpy as np
import gym
import torch
import argparse
import pickle

from functools import partial

from rl.policies import GaussianMLP
from rl.policies import actor

from cassie.cassie_standing_stepping_env import CassieStandingSteppingEnv

def impulse(i):
    if(i == 29):
        return(1)
    else:
        return(0)

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="ID for parallelization")
args = parser.parse_args()

env = partial(CassieStandingSteppingEnv, "stepping", simrate=60, state_est=True)()
policy = torch.load('./trained_models_rl/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt')


state_vector = []
state_labels = []

for i in range(1000):
    state = env.reset("stepping")
    for j in range(200):
        x = np.random.randint(-300, 300)
        y = np.random.randint(-300, 300)
        env.sim.apply_force([x * impulse(j), y * impulse(j), 0, 0, 0, 0])
        if(j == 29):
            print(x, y)
        state = torch.Tensor(state)
        if j > 30 and j <= 80:
            state_vector.append(env.get_full_state("stepping").tolist())
        _, action = policy.act(state, True)
        state, reward, done, info = env.step(np.ndarray.flatten(np.array(action)), "stepping")
        # env.render()
        if j == 199:
            if env.sim.qpos()[2] < 0.75:
                state_labels.append(np.ones(50).tolist())
                print("fail {}".format(i))
            else:
                state_labels.append(np.zeros(50).tolist())
                print("good {}".format(i))

state_vector = np.array(state_vector)
state_labels = np.ndarray.flatten(np.array(state_labels))
v = "state_vector" + str(args.id) + ".p"
l = "state_labels" + str(args.id) + ".p"

pickle.dump(state_vector, open("./data/" + v, "wb"))
pickle.dump(state_labels, open("./data/" + l, "wb"))

env.close()