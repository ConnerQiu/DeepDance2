import torch
import pickle
from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent


#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=True)
cfg.motion_data_file = r"assets\aist_motion\d04_mBR0.pkl"
cfg.render = True #whether create gui window
cfg.model_param_file = 'C:/Users/cq/PycharmProjects/DeepDance/dance.p'

#Create Env and Agent
env = HumanoidEnv(cfg)
agent = Agent(env, cfg)
state = env.reset()

if __name__ == '__main__':
    for i in range(1000):
        traj_batch = agent.sample(10)
        print('sample successfully')
        print(len(agent.memory.memory))
        agent.update_params(traj_batch)
        torch.cuda.empty_cache()

        cp_path = 'C:/Users/cq/PycharmProjects/DeepDance/dance.pkl'
        model_cp = {'policy_dict': agent.policy_net.state_dict(), 'value_dict': agent.value_net.state_dict()}
        pickle.dump(model_cp, open(cp_path, 'wb'))
        print(i,'save successfully')