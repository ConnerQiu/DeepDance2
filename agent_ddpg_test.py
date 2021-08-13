import torch
import pickle
import  time
import os
from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent_DDPG import Agent_DDPG
from utils.memory import Memory

#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=False)
cfg.motion_data_file = r"assets\cmu_mocap\cmu_expert_group.pkl"
cfg.render = False #whether create gui window
cfg.start_iter = 0
cfg.model_param_dir = 'C:/Users/cq/PycharmProjects/DeepDance/results'
cfg.model_param_file = os.path.join(cfg.model_param_dir, "_%s.pkl" % cfg.start_iter)

#Create Env and Agent
if __name__ == '__main__':
    env = HumanoidEnv(cfg)
    agent = Agent_DDPG(env, cfg)

    traj_batch = agent.sample(1500)