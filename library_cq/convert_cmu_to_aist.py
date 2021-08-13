import pickle
import os

from utils.get_expert import get_expert
from utils.config import Config
from envs.Humanoid_Env import HumanoidEnv

#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=False)
cfg.motion_data_file = r"assets\cmu_mocap\cmu_mocap_combine.pkl"
cfg.render = False #whether create gui window
cfg.start_iter = 0
cfg.model_param_dir = 'C:/Users/cq/PycharmProjects/DeepDance/results'
cfg.model_param_file = os.path.join(cfg.model_param_dir, "train_from_cmu_iter_%s.pkl" % cfg.start_iter)

#Create Env and Agent
env = HumanoidEnv(cfg)


save_file = r"C:\Users\cq\PycharmProjects\DeepDance\assets\cmu_mocap\cmu_expert_group.pkl"
data_folder = r"C:\Users\cq\PycharmProjects\RFC\data\cmu_mocap\motion"
expert_meta = {'dt': 0.03333333333333333, 'mocap_fr': 120, 'scale': 0.45, 'offset_z': -0.07, \
            'cyclic': False, 'cycle_offset': 0.0, 'select_start': 0, 'select_end': 176, 'fix_feet': False, 'fix_angle': True}

def convert_cmu_to_aist(data_folder, save_file):
    #get file list
    cmu_file_names = os.listdir(data_folder)
    mujoco_expert_group = []
    for name in cmu_file_names:
        #load data
        cmu_file_path = os.path.join(data_folder, name)
        cmu_data = pickle.load(open(cmu_file_path, "rb"))
        cmu_qpos = cmu_data[0]

        #covert data
        expert = get_expert(cmu_qpos, expert_meta, env)
        expert["name"] = name
        mujoco_expert_group.append(expert) 
        #each expert is a dict wiht keys of "qpos", "rangv" and etc.,
        #mujoco_expert_group is a list
    

    pickle.dump(mujoco_expert_group, open(save_file, "wb"))
    print("save successfully to %s" % save_file)

convert_cmu_to_aist(data_folder, save_file)






    



