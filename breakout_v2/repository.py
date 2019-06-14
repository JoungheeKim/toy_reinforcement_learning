import torch
from collections import deque
from collections import namedtuple
import numpy as np
import random
from skimage.transform import rescale
from skimage.transform import resize
import copy

class memoryDataset(object):
    def __init__(self, maxlen, device):
        self.memory = deque(maxlen=maxlen)
        self.subset = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'life', 'terminal'))


    def push(self, state, action, next_state, reward, done, life, terminal):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        ##False:0, True:1
        done = torch.tensor([done], dtype=torch.long)
        ##Life : 0,1,2,3,4,5
        life = torch.tensor(life, dtype=torch.float)

        terminal = torch.tensor([terminal], dtype=torch.long)
        self.memory.append(self.subset(state, action, reward, next_state, done, life, terminal))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = self.subset(*zip(*batch))
        return batch

class historyDataset(object):
    def __init__(self, history_size, img):
        self.history_size = history_size

        state = self.convert_channel(img)
        self.height, self.width = state.shape

        temp = []
        for _ in range(history_size):
            temp.append(state)
        self.history = temp

    def convert_channel(self, img):
        # input type : |img| = (Height, Width, channel)
        # remove useless item
        img = img[32:193, 8:152]
        #img = rescale(img, 1.0 / 2.0, anti_aliasing=False, multichannel=False)
        img = resize(img, output_shape=(42, 42))

        # conver channel(3) -> channel(1)
        img = np.any(img, axis=2)
        # |img| = (Height, Width)  boolean
        return img

    def push(self, img):
        temp = self.history
        state = self.convert_channel(img)
        temp.append(state)
        self.history = temp[1:]

    def get_state(self):
        #return self.history
        return copy.deepcopy(self.history)
