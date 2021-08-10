import torch
import pickle
import  time
import os
from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent
from utils.memory import Memory

#generate config
cfg_path = "cfg/0506.yml"
cfg = Config(cfg_path, "imitaion", test=False)
cfg.motion_data_file = r"assets\cmu_mocap\cmu_expert_group.pkl"
cfg.render = False #whether create gui window
cfg.start_iter = 0
cfg.model_param_dir = 'C:/Users/cq/PycharmProjects/DeepDance/results'
cfg.model_param_file = os.path.join(cfg.model_param_dir, "train_from_cmu_iter_%s.pkl" % cfg.start_iter)


#Create Env and Agent
env = HumanoidEnv(cfg)
agent = Agent(env, cfg)
state = env.reset()

if __name__ == '__main__':
    for i in range(10000):
        start_time = time.time()
        traj_batch = agent.sample(1500)
        sample_time = time.time() - start_time

        agent.update_params(traj_batch)
        trainging_time = time.time() - start_time - sample_time
        print('sample time: %d, trainging_time: %d in epoch %d' % (sample_time, trainging_time, i))
        torch.cuda.empty_cache()

        if not cfg.test and i%10 == 0:
            cp_path = "%s/train_from_cmu_iter_%s_update_5_singleaction.pkl" % (cfg.model_param_dir, i+cfg.start_iter)
            model_cp = {'policy_dict': agent.policy_net.state_dict(), 'value_dict': agent.value_net.state_dict()}
            pickle.dump(model_cp, open(cp_path, 'wb'))
            print('iteration of %d saved successfully' % i)