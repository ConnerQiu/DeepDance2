import time
import math
import pickle
import os

from typing import Union
from utils.mytorch import to_test, to_train
from utils import batch_to
from utils.mytorch import LongTensor

from agent.model.policy_gaussian import PolicyGaussian
from agent.model.mlp import MLP
from agent.model.critic import Value
from agent.model.logger_rl import LoggerRL
from agent.model.trajbatch import TrajBatch
from envs.Humanoid_Env import HumanoidEnv
from utils import Config
from utils.memory import Memory
from utils.reward_function import reward_func

import numpy as np
import torch
from torch import tensor


class Agent:
    def __init__(self, env: Union[HumanoidEnv, ], cfg: Config):
        #set env, cfg and space
        self.env = env
        self.cfg = cfg
        self.actuators = env.model.actuator_names
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        #set sample params: network, reward function, memory and device
        self.policy_net = PolicyGaussian(MLP(self.state_dim, self.cfg.policy_hsize, self.cfg.policy_htype),
                                         self.action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
        self.value_net = Value(MLP(self.state_dim, self.cfg.value_hsize, self.cfg.value_htype))
        if os.path.exists(cfg.model_param_file):
            model_cp = pickle.load(open(cfg.model_param_file, "rb"))
            self.policy_net.load_state_dict(model_cp['policy_dict'])
            self.value_net.load_state_dict(model_cp['value_dict'])

        self.custom_reward = reward_func[self.cfg.reward_id]
        self.memory = Memory()
        self.traj_cls = TrajBatch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        #other settings
        self.sample_modules = [self.policy_net]
        self.update_modules = [self.policy_net, self.value_net]
        self.render = False
        self.gamma = self.cfg.gamma
        self.tau = 0.95
        self.opt_num_epochs = 1
        self.use_mini_batch = False
        self.dtype = torch.float32
        self.mean_action = True
        self.value_opt_niter = 1
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr, weight_decay=self.cfg.policy_weightdecay)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr, weight_decay=self.cfg.value_weightdecay)
        self.clip_epsilon = 0.2
        self.policy_grad_clip = [(self.policy_net.parameters(), 40)]

        # self.noise_rate = 1.0

    def action(self, state):
        state_var = tensor(state, dtype=torch.float32).unsqueeze(0)
        trans_out = self.trans_policy(state_var)
        mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
        action = self.policy_net.select_action(trans_out, mean_action)[0].detach().numpy()
        action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
        return action

    def sample(self, sample_size):
        total_episode_reward = 0
        for i in range(sample_size):
            state = self.env.reset()
            done = False
            while not done:
                #make action
                action = self.action(state)
                next_state, env_reward, done, info = self.env.step(action) 

                #reward and other calculation
                if self.cfg.custom_reward:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    reward = env_reward
                mask = 0 if done else 1.0
                exp = 1 - self.mean_action
                total_episode_reward += reward

                #save memory and transfer state
                self.push_memory(state, action, mask, next_state, reward, exp)
                state = next_state
            if i%100==0: print('episode %d trained' % (i))
        print('Total reward of this 200 episode is %d'%(total_episode_reward))
        traj_batch = self.traj_cls(self.memory)
        self.meory = Memory()
        return traj_batch

    def trans_policy(self, states):
        """transform states before going into policy net"""
        return states

    def push_memory(self, state, action, mask, next_state, reward, exp):
        self.memory.push(state, action, mask, next_state, reward, exp)

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states))

        """get advantage estimation from the trajectories"""
        advantages, returns = self.estimate_advantages(rewards, masks, values, self.gamma, self.tau)
        print('calculate advantages successful!')

        self.update_policy(states, actions, returns, advantages, exps)
        print('training done!')

        return time.time() - t0
    
    def estimate_advantages(self, rewards, masks, values, gamma, tau):
        device = rewards.device
        rewards, masks, values = batch_to(torch.device('cpu'), rewards, masks, values)
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)

        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        advantages, returns = batch_to(device, advantages, returns)
        return advantages, returns

    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(self.trans_value(states))
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = LongTensor(perm).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs, ind)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss

    def trans_value(self, states):
        return states    
