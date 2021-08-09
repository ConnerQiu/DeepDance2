import torch
import pickle
import  time
from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent
from utils.memory import Memory

#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=False)
cfg.motion_data_file = r"assets\aist_motion\d04_mBR0.pkl"
cfg.render = False #whether create gui window
cfg.model_param_file = 'C:/Users/cq/PycharmProjects/DeepDance/d04_mBR0.pkl'

#Create Env and Agent
env = HumanoidEnv(cfg)
agent = Agent(env, cfg)
state = env.reset()

if __name__ == '__main__':
    for i in range(10000):
        start_time = time.time()
        traj_batch = agent.sample(10)
        sample_time = time.time() - start_time

        # for _ in range (100):
        agent.update_params(traj_batch)
        trainging_time = time.time() - start_time - sample_time
        print('sample time: %d, trainging_time: %d' % (sample_time, trainging_time))
        torch.cuda.empty_cache()
        cp_path = 'C:/Users/cq/PycharmProjects/DeepDance/d04_mBR0.pkl'
        model_cp = {'policy_dict': agent.policy_net.state_dict(), 'value_dict': agent.value_net.state_dict()}
        pickle.dump(model_cp, open(cp_path, 'wb'))
        print(i,'save successfully')