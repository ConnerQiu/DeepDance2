import multiprocessing
import time
import os
from utils.config import Config
from envs.Humanoid_Env import HumanoidEnv
from pathlib import Path
from os import path
import mujoco_py

class TestAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.num_threads = 8

    def sample_worker(self, pid, queue):
        print('it is just a test, from prcess %d' % pid)
        queue.put([pid,time.time()])

    def sample(self,):
        queue = multiprocessing.Queue()
        process_list = []
        num_threads = self.num_threads
        thread_batch_size =12
        for i in range(num_threads-1):
            worker_args = (i, queue)
            worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
            worker.start()
            process_list.append(worker)

        for i in process_list:
            worker.join()

        for i in range(num_threads-2):
            pin, time = queue.get()
            print(pin, time)

class  TestEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        if not path.exists(cfg.mujoco_model_file):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent, 'DeepDance/assets/mujoco_models', path.basename(cfg.mujoco_model_file))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        # self.model = mujoco_py.load_model_from_path(fullpath)
        # self.sim = mujoco_py.MjSim(self.model)
        print('create successfully')
        


if __name__ == '__main__':
    #generate config
    cfg_path = "cfg/0506.yml"
    cfg = Config(cfg_path, "imitaion", test=False)
    cfg.motion_data_file = r"assets\cmu_mocap\cmu_expert_group.pkl"
    cfg.render = False #whether create gui window
    cfg.start_iter = 0
    cfg.model_param_dir = 'C:/Users/cq/PycharmProjects/DeepDance/results'
    cfg.model_param_file = os.path.join(cfg.model_param_dir, "_%s.pkl" % cfg.start_iter)
    env = TestEnv(cfg)

    agent = TestAgent(env, cfg)
    agent.sample()