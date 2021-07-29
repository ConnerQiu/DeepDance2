import numpy as np


class TrajBatch:

    def __init__(self, memory):

        self.batch = zip(*memory.sample())
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
