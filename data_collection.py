import numpy as np
import gym
import torch
import argparse
import pickle

from functools import partial

from rl.policies import GaussianMLP
from rl.policies import actor

from cassie.cassie_standing_env import CassieStandingEnv

def impulse(i):
    if(i == 29):
        return(1)
    else:
        return(0)

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="ID for parallelization")
args = parser.parse_args()

env = partial(CassieStandingEnv, "stepping", simrate=60, state_est=True)()
policy = torch.load('./trained_models_rl/standing_policy_humanoid_paper.pt')


state_vector = []
state_labels = []

for i in range(1000):
    state = env.reset()
    for j in range(200):
        x = np.random.randint(-250, 250)
        y = np.random.randint(-250, 250)
        env.sim.apply_force([x * impulse(j), y * impulse(j), 0, 0, 0, 0])
        if(j == 29):
            print(x, y)
        # env.sim.step_pd(pd_in_t())
        state = torch.Tensor(state)
        if j > 30 and j <= 80:
            state_vector.append(env.get_full_state().tolist())
        action = policy.act(state, True)
        state, reward, done, info = env.step(np.ndarray.flatten(np.array(action)))
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