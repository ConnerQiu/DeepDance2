import torch
import pickle
import  time
import os
import numpy as np

from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent
from utils.memory import Memory


#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=False)
cfg.motion_data_file = r"assets\cmu_mocap\cmu_expert_group.pkl"
cfg.render = False #whether create gui window

env = HumanoidEnv(cfg)
env.reset

for i in range (10):
    print(env.cur_expert_num, env.cur_t, env.start_ind)
    cmu_pose = env.get_expert_qpos_test()
    action = np.zeros(38)
    action[0:32] = cmu_pose[7:]
    action[32:35] = cmu_pose[0:3]
    action[35:38] = cmu_pose[4:7]
    action = np.array([ 0.02176815,  0.00051212,  0.01892708,  0.01286933,  0.00039439,
       -0.00865127, -0.00114013,  0.00960663, -0.00438942, -0.00057183,
        0.00721685, -0.00259421, -0.00420993, -0.00058979, -0.01642344,
       -0.00752245, -0.00410568, -0.01283858, -0.01460044,  0.02084938,
        0.00753419,  0.00374878,  0.0074125 ,  0.00074201,  0.00698855,
       -0.01076522,  0.00080448,  0.00993106,  0.01310821, -0.00380814,
        0.01559255,  0.00201797, -0.01678168, -0.00767266,  0.01410458,
        0.01519752,  0.00442936, -0.00737093])

    next_state, env_reward, done, info = env.step(action)
    print(env_reward, i)