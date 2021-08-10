import torch
import pickle
from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent


#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=True)
cfg.motion_data_file = r"assets\aist_motion\d04_mBR0.pkl"
cfg.render = False #whether create gui window
cfg.model_param_file = 'C:/Users/cq/PycharmProjects/DeepDance/dance.p'

#Create Env and Agent
env = HumanoidEnv(cfg)
env.reset()
obs = env.get_obs()
print(obs)
# expert_qpos, expert_meta = pickle.load(open(r"C:\Users\cq\PycharmProjects\RFC\data\cmu_mocap\motion\05_06.p", "rb"))
# print(expert_qpos.shape, expert_meta)
