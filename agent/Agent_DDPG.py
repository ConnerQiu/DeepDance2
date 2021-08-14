import copy
import time
import torch
import pickle
import numpy as np
import os
import multiprocessing

from torch import tensor

from agent.model.network import *
from utils.memory import Memory
from utils.reward_function import reward_func

class Agent_DDPG:
    def __init__(self,env, cfg ) -> None:
        self.env = env 
        self.cfg = cfg
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]-6

        """"set up policy and value network"""
        self.policy_net = PolicyDetermin(self.state_dim, self.cfg.policy_hsize, self.action_dim)
        self.value_net = Value(self.state_dim, self.cfg.value_hsize)
        if os.path.exists(cfg.model_param_file): #load params from previous training results
            model_cp = pickle.load(open(cfg.model_param_file, 'rb'))
            self.policy_net.load_state_dict(model_cp['policy_dict'])
            self.value_net.load_state_dict(model_cp['value_dict'])
            print('load params from %s successfully' % cfg.model_param_file)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.target_value_net = copy.deepcopy(self.value_net)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), \
            lr=self.cfg.policy_lr, weight_decay=self.cfg.policy_weightdecay)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), \
            lr=self.cfg.value_lr, weight_decay=self.cfg.value_weightdecay)

        self.num_threads = 2
        self.custom_reward = reward_func[self.cfg.reward_id]
        print('create agent successfully')

    def action(self, state):
        state_var = tensor(state, dtype=torch.float32).unsqueeze(0)
        trans_out = self.trans_policy(state_var)
        action = self.policy_net.select_action(trans_out)[0].detach().numpy()
        action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
        return action

    def trans_policy(self, state):
        return state

    def sample_worker(self, queue):
        print('it is just a test, from prcess %d' % pid)
        # queue.put([pid, thread_batch_size,time.time()])

    def sample_worker_back(self, pid, queue, thread_batch_size):
        torch.randn(pid)
        memory = Memory()
        for i in range(thread_batch_size):
            state = self.env.reset()
            done = False
            while not done :
                action = self.action(state)
                next_state, env_reward, done, info = self.env.step(action)
                if self.cfg.custom_reward:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    reward = env_reward
                mask = 0 if done else 1.0
                exp = 1
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                state = next_state

                #render GUI
                if pid==0 and self.cfg.render:
                    self.env.render()
            
            if i%50==0:
                print('worker %d has sampled %d episodes' %(pid, i))
        queue.put([pid, memory])

    def sample(self, min_batch_size):
        print('start sample')
        start_time = time.time()
        with torch.no_grad():
            thread_batch_size = int(min_batch_size / self.num_threads)
            queue = multiprocessing.Queue()
            process_list = []
            memories = [None] * self.num_threads
            for i in range(self.num_threads-1):
                # worker_args = (i+1, queue, thread_batch_size)
                worker = multiprocessing.Process(target=self.sample_worker, args=(queue,))
                worker.start()
                process_list.append(worker)
            
            for i in process_list:
                worker.join()
            # memories[0] = self.sample_worker(0, None, thread_batch_size)

            # for i in range(self.num_threads-1):
            #     pin, worker_memory = queue.get()
            #     memories[pid] = worker_memory
            # # traj_batch = self.traj_cls(memories)
        sample_time = time.time()-start_time
        print('sample end, takes %d seconds in total' % sample_time)

        # return traj_batch

    def push_memory(self, memory, state, action, mask, next_state, reward, exp):
        memory.push(state, action, mask, next_state, reward, exp)
