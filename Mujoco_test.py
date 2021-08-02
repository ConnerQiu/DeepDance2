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



qpos = [0., 0.,  1.,  0,  0, 0.,  -0.0,  0.,  0.,  -0.,  0.0,  0,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.0,  0,  0.,  0.,  0.0,  0.0,  0.0,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
qpos = numpy.array(qpos)
print(qpos.shape)
qvel = sim.data.qvel

set_state(qpos,qvel)
viewer = mujoco_py.MjViewer(sim)

for i in range(1000000):
    viewer.render()