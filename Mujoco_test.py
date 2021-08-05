from Cython.Utils import long_literal
import mujoco_py
import numpy

def set_state(qpos, qvel):
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)

model = mujoco_py.load_model_from_path('assets/mujoco_models/mocap_v2.xml')
sim = mujoco_py.MjSim(model)
data = sim.data

def set_base_pos():
    qpos_base = qpos.copy()
    qpos_base[7] = -0.3490658503988659
    qpos_base[10] = 0.7853981633974483
    qpos_base[14] = 0.3490658503988659
    qpos_base[17] = 0.7853981633974483
    qpos_base[29] = 1.3962634015954636
    qpos_base[32] = 0.7853981633974483
    qpos_base[35] = -1.3962634015954636
    qpos_base[38] = 0.7853981633974483
    return qpos_base

qpos = [0., 0.,  1.,  0,  0, 0.,  -0.0,  0.,  0.,  -0.,  0.0,  0,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.0,  0,  0.,  0.,  0.0,  0.0,  0.0,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
qpos_base = set_base_pos()

qpos = numpy.array(qpos_base)
print(qpos.shape)
qvel = sim.data.qvel

set_state(qpos,qvel)
viewer = mujoco_py.MjViewer(sim)

for i in range(1000000):
    viewer.render()