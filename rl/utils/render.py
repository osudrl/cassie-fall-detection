import torch
from torch.autograd import Variable
import time
import numpy as np
import sys
import select
import tty
import termios

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    env.speed = 1
    env.phase_add = 2
#     env.sim.set_qpos([0.000000, 0.000000, 1.010000, 1.000000, 0.000000, 0.000000, 0.000000, 0.0000, 0.000000, 0.00, 0.978483, 
#     -0.016400, 0.017870, -0.204896, 00, 0.000000, 1.426700, 0.000000, -1.524400, 1.524400, 00, -0.0000, 0.000000, 
#     0.0, 0.978614, 0.003860, -0.015240, -0.205103, 00, 0.000000, 1.426700, 0.000000, -1.524400, 1.524400, 00])

    render_state = env.render()
    print("render_state:", render_state)
    while render_state:
        if (not env.vis.ispaused()):
            _, action = policy.act(state, deterministic)
            if deterministic:
                action = action.data.numpy()
            else:
                action = action.data[0].numpy()
            # print("action:", action)
            # print("action: ", action.data.numpy())

            state, reward, done, _ = env.step(action)
            print("speed: ", env.sim.qvel()[0])

            # if done:
            #     state = env.reset()

            state = torch.Tensor(state)

        render_state = env.render()
        time.sleep(dt / speedup)

def avg_pols(policies, state, deterministic):
    total_act = np.zeros(10)
    for policy in policies:
        _, action = policy.act(state, deterministic)
        if deterministic:
            action = action.data.numpy()
        else:
            action = action.data[0].numpy()
        total_act += action
    return total_act / len(policies)

@torch.no_grad()
def rendermultipolicy(env, policies, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    env.speed = 1
    env.phase_add = 1

    render_state = env.render()
    print("render_state:", render_state)
    while render_state:
        if (not env.vis.ispaused()):
            action = avg_pols(policies, state, deterministic)
            # print("action:", action)

            state, reward, done, _ = env.step(action)

            # if done:
            #     state = env.reset()

            state = torch.Tensor(state)

        render_state = env.render()
        time.sleep(dt / speedup)

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

@torch.no_grad()
def renderpolicy_speedinput(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    env.speed = .4
    env.phase_add = 2

    render_state = env.render()
    old_settings = termios.tcgetattr(sys.stdin)
    print("render_state:", render_state)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while render_state:
            if isData():
                c = sys.stdin.read(3)
                if c == '\x1b[A':
                    env.speed += .1
                    print("Increasing speed to: ", env.speed)
                elif c == '\x1b[B':
                    env.speed -= .1
                    print("Decreasing speed to: ", env.speed)
                else:
                    pass
            if (not env.vis.ispaused()):
                _, action = policy.act(state, deterministic)
                if deterministic:
                    action = action.data.numpy()
                else:
                    action = action.data[0].numpy()
                # print("action:", action)
                # print("action: ", action.data.numpy())

                state, reward, done, _ = env.step(action)
                print("speed: ", env.sim.qvel()[0])

                # if done:
                #     state = env.reset()

                state = torch.Tensor(state)

            render_state = env.render()
            time.sleep(dt / speedup)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)