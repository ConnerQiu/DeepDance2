import pickle
import os
import torch
import numpy as np
import mujoco_py
import copy

"""load pose data from smpl file"""
def load_smpl_motion(motion_file):

    f = open(motion_file, 'rb')
    full_data = pickle.load(f)
    pose_data = full_data['smpl_poses']
    position_data = full_data['smpl_trans']
    position_data[:,0] = position_data[:,0]/20
    position_data[:,1] = position_data[:,1]/340
    position_data[:,2] = position_data[:,1]/120+0.85

    pose_data = np.concatenate((pose_data, position_data), axis=1)
    smpl_tensors = torch.tensor(pose_data)
    return smpl_tensors

"""the matching relatinship between mujoco models and smpl models"""
qpos_key = {0:'root-x', 1:'root-y', 2:'root-z', 3:'root', 4:'root-rx', 5:'root-ry', 6:'root-rz',\
    7:'lfemur-x', 8:'lfemur-y', 9:'lfemur-z', 10:'ltibia-x',11:'lfoot-x', 12:'lfoot-y', 13:'lfoot-z',\
    14:'rfemur-x', 15:'rfemur-y', 16:'rfemur-z', 17:'rtibia-x', 18:'rfoot-x', 19:'rfoot-y', 20:'rfootz', \
    21:'upperback-x', 22:'upperback-y', 23:'upperback-z', 24:'lowerneck-x', 25:'lowerneck-y', 26:'lowerneck-z',\
    27:'lclavicle-x', 28:'lclavicle-y', 29:'lhumerus-x', 30:'lhumerus-y', 31:'lhumerus-z', 32:'lradius',\
    33:'rclavicle-x', 34:'rclavicle-y', 35:'rhumerus-x', 36:'rhumerus-y', 37:'rhumerus-z', 38:'rradius'}

smpl_key = {'root-x': 73, 'root-y': 72, 'root-z': 74, 'root': 0, 'root-rx':-0, 'root-ry':-2, 'root-rz':1,\
    'lfemur-x': 5, 'lfemur-y': 4, 'lfemur-z': 3, 'ltibia-x': 12, 'lfoot-x': 22, 'lfoot-y': 23, 'lfoot-z': 21, \
    'rfemur-x': 8, 'rfemur-y': 7, 'rfemur-z': 6, 'rtibia-x': 15, 'rfoot-x': 25, 'rfoot-y': 26, 'rfootz': 24, \
    'upperback-x': 20, 'upperback-y': 19, 'upperback-z': 18, 'lowerneck-x': 29, 'lowerneck-y': 28, 'lowerneck-z': 27, \
    'lclavicle-x': 41, 'lclavicle-y': 40, 'lhumerus-x': -50, 'lhumerus-y': -48, 'lhumerus-z': -49, 'lradius': -55, \
    'rclavicle-x': 44, 'rclavicle-y': 42, 'rhumerus-x': -53, 'rhumerus-y': -51, 'rhumerus-z': -52, 'rradius': 58}

base_pose_params = torch.tensor([0.0, 0.0, 0.0, 0., 0., 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.57, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

"""translate smpl pose to mujoco pose"""
def get_pose(smpl_pose):
    pose = np.zeros(39)
    for i in range(len(pose)):
        smpl_num = smpl_key[qpos_key[i]]
        if smpl_num == 5:
            pose[i] = smpl_pose[smpl_num]-0.3
        elif smpl_num == 8:
            pose[i] = smpl_pose[smpl_num]+0.3
        else:
            pose[i] = smpl_pose[smpl_num] if smpl_num>0 else -smpl_pose[abs(smpl_num)]
    return pose

def set_state(qpos, qvel):
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)

def set_smpl_pose(i, k):
    my_pose_params = copy.deepcopy(base_pose_params)
    my_pose_params[i] = 3.14 * ((k)/(20/2))
    return my_pose_params

def tranlate_smpl_to_mujoco(smpl_tensors):
    num_frames = smpl_tensors.size()[0]
    mujoco_poses = np.zeros(shape=(num_frames, 39))
    for i in range(num_frames):
        mujoco_poses[i] = get_pose(smpl_tensors[i])  
    return mujoco_poses

def visualize(expert_traj, sim):

    """render or select part of the clip"""
    viewer = mujoco_py.MjViewer(sim)
    viewer._hide_overlay = True
    T = 10
    fr = 0
    paused = False
    stop = False

    # viewer.custom_key_callback = key_callback
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -8.0
    viewer.cam.distance = 5.0
    viewer.cam.lookat[2] = 1.0
    t = 0
    while not stop:
        if t >= T:
            fr = (fr+1) % expert_traj.shape[0]
            t = 0
        sim.data.qpos[:] = expert_traj[fr]
        sim.data.qpos[2] = 0.85
        sim.data.qpos[3] = 0
        sim.data.qpos[4] = 0
        sim.data.qpos[5] = 0
        print(sim.data.qpos[0:7])

        sim.forward()
        viewer.cam.lookat[:2] = sim.data.qpos[:2]
        viewer.render()
        if not paused:
            t += 1
    viewer.save_

def frame_visulize(expert_traj, show_frames):
    sim.data.qpos[:] = expert_traj
    sim.data.qpos[2] = 0.85
    sim.data.qpos[3] = 0
    viewer = mujoco_py.MjViewer(sim)
    viewer.cam.lookat[:2] = sim.data.qpos[:2]
    # viewer.cam.azimuth = 45
    viewer.cam.elevation = -8.0
    viewer.cam.distance = 5.0
    # viewer.cam.lookat[2] = 1.0
    for i in range(show_frames):
        viewer.render()
        print(sim.data.qpos[4], i)

def test_single_pose(mujoco_poses, a, b):
    for i in range(a, b):
        # mujoco_poses[5*i, 0] = 0
        # mujoco_poses[5*i, 1] = 0
        # mujoco_poses[5*i, 2] = 0
        # mujoco_poses[5*i, 3] = 0
        mujoco_poses[5*i, 4] = 0.00001
        mujoco_poses[5*i, 5] = 0.00001
        # mujoco_poses[5*i, 6] = 0
        
        frame_visulize(mujoco_poses[5*i], 100)
        print(5*i, mujoco_poses[5*i])
        print(sim.data.qpos)

def test_base_model_transalation():
    """test the tranlation effect"""
    #build env
    model = mujoco_py.load_model_from_path('assets/mujoco_models/mocap_v2.xml')
    sim = mujoco_py.MjSim(model)
    data = sim.data

    # set state
    data = load_smpl_motion('gBR_sBM_cAll_d04_mBR0_ch02.pkl')
    # data[0] = 0
    # qpos = get_pose(data)

    test_pose_params = set_smpl_pose(0,-5)
    qpos = get_pose(test_pose_params)
    qpos[0] = 0.
    qpos[1] = 0.0
    qpos[2] = 0.91
    qpos[3] = 1.57

    qvel = sim.data.qvel
    set_state(qpos,qvel)
    viewer = mujoco_py.MjViewer(sim)
    print(viewer.sim.data.qpos)
    print('which script')

    #simulate
    for i in range(1000000):
        viewer.render()

def translate_and_save(smpl_file, render):
    model_folder = 'assets/aist_motion'
    smpl_tensors = load_smpl_motion(smpl_file)
    mujoco_poses = tranlate_smpl_to_mujoco(smpl_tensors)
    save_path = os.path.join(model_folder,os.path.splitext(smpl_file)[0]+'.p')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(mujoco_poses, open(save_path,'wb'))
    print('Successfully saved mujoco_poses to %s !.'% save_path)

    if render:
        model = mujoco_py.load_model_from_path('assets/mujoco_models/mocap_v2.xml')
        sim = mujoco_py.MjSim(model)
        visualize(mujoco_poses, sim)


#test load whole dacnce motion
# smpl_file = 'gBR_sBM_cAll_d05_mBR0_ch01.pkl'
# translate_and_save(smpl_file, True)

# model = mujoco_py.load_model_from_path('assets/mujoco_models/mocap_v2.xml')
# sim = mujoco_py.MjSim(model)

def translate_expert_group(save_name):
    model_folder = r'C:\Users\cq\AIST_data\AIST_annotation\motions'
    motion_list = os.listdir(model_folder)
    expert_group = []
    for motion_name in motion_list[0:9]:
        smpl_file = os.path.join(model_folder,motion_name)
        smpl_tensors = load_smpl_motion(smpl_file)
        mujoco_poses = tranlate_smpl_to_mujoco(smpl_tensors)
        expert = {}
        expert["qpos"] = mujoco_poses
        expert["name"] = motion_name
        expert_group.append(expert)

    save_path = "assets/aist_motion/"+save_name+".pkl"
    print(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(expert_group, open(save_path,'wb'))
    print('Successfully saved mujoco_poses to %s !.'% save_path)

translate_expert_group("d04_mBR0")

full_data = pickle.load(open("assets/aist_motion/test.pkl", 'rb'))
print(full_data[7]["name"])
