from envs.Humanoid_Env import HumanoidEnv
from utils.config import Config
from agent.Agent import Agent

#generate config
cfg = Config(cfg_id='0506', test=False, create_dirs=not (True or 1000 > 0))
cfg.render = False

#Create Env and Agent
env = HumanoidEnv(cfg)
agent = Agent(env, cfg)
state = env.reset()

if __name__ == '__main__':
    for i in range(10):
        traj_batch = agent.sample(200)
        agent.update_params(traj_batch)
        torch.cuda.empty_cache()