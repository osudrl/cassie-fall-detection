import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

from rl.policies import GaussianMLP
from rl.policies import actor

from cassie.cassie_standing_stepping_env import CassieStandingSteppingEnv

env = partial(CassieStandingSteppingEnv, "stepping", simrate=60, state_est=True)()

policy_step = torch.load('./trained_models_rl/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt')
policy_stand = torch.load('./trained_models_rl/standing_policy_humanoid_paper.pt')

model = pickle.load(open("./trained_models_sl/policy_selector.p", "rb" ))

def impulse(i):
    if(i == 400):
        return(1)
    else:
        return(0)

factor = 10

state = env.reset('standing')
for i in range(10 ** 10):
    env.sim.apply_force([200 * impulse(i), 200 * impulse(i), 0, 0, 0, 0])
    state = torch.Tensor(state)
    # print(model(state[0:46].unsqueeze(0).float()))
    if torch.norm(model(state[0:46].unsqueeze(0).float())) > 0.5:
        _, action = policy_step.act(torch.from_numpy(env.get_full_state('stepping')).float(), deterministic = True)
        state, reward, done, info = env.step(action.data.numpy(), 'stepping')
    else:
        action = policy_stand.act(torch.from_numpy(env.get_full_state('standing')).float(), deterministic=True)
        state, reward, done, info = env.step(action.data.numpy(), 'standing')
    env.render()
    # if i == 0:
    #     input()
    
env.close()