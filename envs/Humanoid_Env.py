from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from pathlib import Path
from mujoco_py import functions as mjf
import mujoco_py
import pickle
import time
from scipy.linalg import cho_solve, cho_factor

from utils.mjviewer import MjViewer
from utils.mujoco import get_body_qposaddr
from utils.transformation import quaternion_from_euler
from utils.get_expert import get_expert
from utils.math import *

DEFAULT_SIZE = 500

class HumanoidEnv:

    def __init__(self, cfg):
        self.cfg = cfg

        #load model and create mujoco env
        if not path.exists(cfg.mujoco_model_file):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent, 'assets/mujoco_models', path.basename(cfg.mujoco_model_file))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = cfg.frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.set_model_params()
        self.set_cam_first = set()
        self.np_random = None
        self.seed()


        #pose related data
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.prev_qpos = None
        self.prev_qvel = None
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None

        #load expert group
        self.expert_group = pickle.load(open(cfg.motion_data_file, "rb"))
        self.cur_expert = None
        self.cur_expert_num = 0
        self.load_expert(self.cur_expert_num)


        #set dim and space
        self.obs_dim = None
        self.action_space = None
        self.observation_space = None
        self.start_ind = 0

        self.cur_t = 0  # number of steps taken
        self.set_spaces()

        # env specific
        self.end_reward = 0.0

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        print("create successfully")

    #initial setup--------------

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # print(self.np_random)
        return [seed]

    def set_model_params(self):
        if self.cfg.action_type == 'torque' and hasattr(self.cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

    def covert_expert(self, expert_num):

        expert_qpos = self.expert_group[expert_num]["qpos"]
        expert_meta = {'dt': 0.03333333333333333, 'mocap_fr': 120, 'scale': 0.45, 'offset_z': -0.07, \
            'cyclic': False, 'cycle_offset': 0.0, 'select_start': 0, 'select_end': 176, 'fix_feet': False, 'fix_angle': True}
        self.cur_expert = get_expert(expert_qpos, expert_meta, self)
        # if the expert group is only qpos, need this function to convert the expert to a proper structure

    def load_expert(self, expert_num):
        self.cur_expert = self.expert_group[expert_num]

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = self.model.body_names[1:]
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.action_dim = self.ndof + self.vf_dim
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'image':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it, and the image format is BGR for OpenCV
            return data[::-1, :, [2, 1, 0]]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = MjViewer(self.sim)
            elif mode == 'image':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self._viewers[mode] = self.viewer
        self.viewer_setup(mode)
        return self.viewer

    #get information---------

    def get_obs(self):
        if self.cfg.obs_type == 'full':
            obs = self.get_full_obs()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # cur_pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # cur_vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full':
            obs.append(qvel)
        
        #get expert pos
        expert_qpos = self.get_expert_qpos()
        obs.append(expert_qpos)

        # phase
        # if self.cfg.obs_phase:
        #     phase = self.get_phase()
        #     obs.append(np.array([phase]))
        obs = np.concatenate(obs)
        return obs

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_com(self):
        return self.data.subtree_com[0, :].copy()

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_expert_qpos(self):
        cur_index = self.get_expert_index(self.cur_t)
        expert_qpos = self.cur_expert["qpos"][cur_index][7:] 
        #self.cur_t means how many steps this episode has taken; 
        #start_ind+1 means the start pose of cur_expert. together it means the next pose it is going to learn
        #only make the joint angles as input
        return expert_qpos

    def get_expert_qpos_test(self):
        cur_index = self.get_expert_index(self.cur_t)
        expert_qpos = self.cur_expert["qpos"][cur_index][:] 
        #self.cur_t means how many steps this episode has taken; 
        #start_ind+1 means the start pose of cur_expert. together it means the next pose it is going to learn
        #only make the joint angles as input
        return expert_qpos    

    def get_expert_index(self, t):
        return ((self.start_ind+t+1) % self.cur_expert['len'])

    def get_expert_attr(self, attr, ind):
        return self.cur_expert[attr][ind, :]

    """methods to be deleted"""
    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    #env control-------------------------------

    def reset(self):
        self.sim.reset()
        self.cur_t = 0
        self.cur_expert_num = 4
        self.load_expert(self.cur_expert_num)
        ob = self.reset_model()
        old_viewer = self.viewer
        for mode, v in self._viewers.items():
            self.viewer = v
            self.viewer_setup(mode)
        self.viewer = old_viewer
        # print("reset successfully, current expert is no %d" % self.cur_expert_num)
        return ob

    def reset_model(self):
        self.start_ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.cur_expert['len'])
        cfg = self.cfg
        if self.cur_expert is not None:
 
            ind = self.start_ind
            # print("start from pose %d" % ind)
            init_pose = self.cur_expert['qpos'][ind, :].copy()
            init_vel = self.cur_expert['qvel'][ind, :].copy()
            init_pose[:7] = [0., 0., 0.85, 0., 0., 0., 0.,] #confirm the impact
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            # print(init_pose, "set init pose")
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            # print(self.sim.data.qpos, "setted init pose")
            # self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0:
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0:
                expert['cycle_relheading'] = quaternion_multiply(get_heading_q(self.data.qpos[3:7]),
                                                              quaternion_inverse(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.data.qpos[:2], expert['init_pos'][[2]]))

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()



    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy()
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        k_p[6:] = cfg.jkp
        k_d[6:] = cfg.jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        return torque

    """ RFC-Explicit """
    def rfc_explicit(self, vf):
        qfrc = np.zeros_like(self.data.qfrc_applied)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            contact_point = vf[i*self.body_vf_dim: i*self.body_vf_dim + 3]
            force = vf[i*self.body_vf_dim + 3: i*self.body_vf_dim + 6] * self.cfg.residual_force_scale
            torque = vf[i*self.body_vf_dim + 6: i*self.body_vf_dim + 9] * self.cfg.residual_force_scale if self.cfg.residual_force_torque else np.zeros(3)
            contact_point = self.pos_body2world(body, contact_point)
            force = self.vec_body2world(body, force)
            torque = self.vec_body2world(body, torque)
            mjf.mj_applyFT(self.model, self.data, force, torque, contact_point, body_id, qfrc)
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.data.qfrc_applied[:vf.shape[0]] = vf

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            if cfg.action_type == 'position':
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)

            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.cur_t += 1
        self.bquat = self.get_body_quat() 
        # self.update_expert()
        # get obs
        head_pos = self.get_body_com('head')
        reward = 1.0
        if cfg.env_term_body == 'head':
            fail = self.cur_expert is not None and head_pos[2] < self.cur_expert['head_height_lb'] - 0.1
        else:
            fail = self.cur_expert is not None and self.data.qpos[2] < self.cur_expert['height_lb'] - 0.1
        cyclic = self.cur_expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.cur_expert['len'] + cfg.env_expert_trail_steps)
        done = fail or end
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end}


