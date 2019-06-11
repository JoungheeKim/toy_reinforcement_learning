import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
import random
from collections import namedtuple
from PIL import Image, ImageDraw
import os
from argparse import ArgumentParser
from tqdm import tqdm
import logging

##LOGGING PROPERTY
LOG_FILE = 'logfile'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--device", dest="device", metavar="device", default="gpu")
    parser.add_argument("--env",dest="env", metavar="env", default="BreakoutDeterministic-v4")
    parser.add_argument("--memory_size", dest="memory_size", metavar="memory_size", default=1000000)
    parser.add_argument("--update_freq", dest="update_freq", metavar="update_freq", default=4)
    parser.add_argument("--learn_start", dest="learn_start", metavar="learn_start", default=50000)
    parser.add_argument("--history_size", dest="history_size", metavar="history_size", default=2)


    ##Learning rate
    parser.add_argument("--batck_size", dest="batck_size", metavar="batck_size", default=32)
    parser.add_argument("--ep", dest="ep", metavar="ep", default=1)
    parser.add_argument("--eps_end", dest="eps_end", metavar="eps_end", default=0.1)
    parser.add_argument("--eps_endt", dest="eps_endt", metavar="eps_endt", default=1000000)
    parser.add_argument("--lr", dest="lr", metavar="lr", default=0.00025)
    parser.add_argument("--discount", dest="discount", metavar="discount", default=0.99)


    parser.add_argument("--agent_type", dest="agent_type", metavar="agent_type", default="DQN_ln")
    parser.add_argument("--max_steps", dest="max_steps", metavar="max_steps", default=50000000)
    parser.add_argument("--eval_freq", dest="eval_freq", metavar="eval_freq", default=250000)
    parser.add_argument("--eval_steps", dest="eval_steps", metavar="eval_steps", default=125000)
    return parser


class DQN_ln(nn.Module):
    def __init__(self, init_dim, layers, class_num, device):
        super(DQN_ln, self).__init__()
        input_dim = init_dim
        self.device = device
        linear_layers = nn.ModuleList()
        for dim in layers:
            linear_layers.append(nn.Linear(input_dim, dim).to(device))
            linear_layers.append(nn.LeakyReLU().to(device))
            input_dim = dim
        linear_layers.append(nn.Linear(input_dim, class_num).to(device))
        self.linear_layers = linear_layers

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x

class memoryDataset(object):
    def __init__(self, maxlen, device):
        self.memory = deque(maxlen=maxlen)
        self.subset = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'life'))
        self.device = device

    def push(self, state, action, next_state, reward, done, life):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor([action], dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        ##False:0, True:1
        done = torch.tensor([done], dtype=torch.long).to(self.device)
        ##Life : 0,1,2,3,4,5
        life = torch.tensor(life, dtype=torch.float).to(self.device)
        self.memory.append(self.subset(state, action, reward, next_state, done, life))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = self.subset(*zip(*batch))
        return batch

class historyDataset(object):
    def __init__(self, history_size, img, device):
        self.history_size = history_size
        self.device = device

        state = self.convert_channel(img)
        self.height, self.width = state.shape

        temp = []
        for _ in range(history_size):
            temp.append(state)
        self.history = temp

    def convert_channel(self, img):
        # input type : |img| = (Height, Width, channel)
        # remove useless item
        img = img[32:190, 8:152]

        # conver channel(3) -> channel(1)
        img = np.any(img, axis=2)
        # |img| = (Height, Width)  boolean
        return img

    def push(self, img):
        temp = self.history
        state = self.convert_channel(img)
        temp.append(state)
        self.history = temp[1:]

    def get_import_value(self, imgs):
        output = np.array([])

        # Set Bar of last Picture
        bar_mean = np.mean(np.where(imgs[-1][-1])) / self.width
        output = np.append(output, bar_mean)  ##bar index

        # Set Block that still exist
        temp = [1]
        for img in imgs:
            temp = temp * img
        output = np.append(output, temp[25:61].reshape(-1))

        # Find Ball Position
        for img in imgs:
            diff = np.where(np.diff([img[:-1] > temp[:-1]]))
            if len(diff[0]) > 0:
                output = np.append(output, np.mean(diff[1]) / self.height)
                output = np.append(output, np.mean(diff[2]) / self.width)
            else:
                output = np.append(output, np.array([0]))
                output = np.append(output, np.array([0]))

        return output

    def get_state(self):
        return self.get_import_value(self.history)

class DQNSolver():

    def __init__(self, config):
        self.device = config.device
        self.env = gym.make(config.env)
        self.memory_size = config.memory_size
        self.update_freq = config.update_freq
        self.learn_start = config.learn_start
        self.history_size = config.history_size


        self.batck_size = config.batck_size
        self.ep = config.ep
        self.eps_end = config.eps_end
        self.eps_endt = config.eps_endt
        self.lr = config.lr
        self.discount = config.discount

        self.agent_type = config.agent_type
        self.max_steps = config.max_steps
        self.eval_freq = self.eval_freq
        self.eval_steps = self.eval_steps


        ##Breakout Setting
        self.layers = [1000,200]
        self.class_num = 4
        self.init_dim = 5185 + (2 * config.history_size)
        self.resize_unit = (161, 144)

        ##INIT SETTING
        self.memory = memoryDataset(maxlen=config.memory_size, device=config.device)
        self.model = DQN_ln(self.init_dim, self.layers, self.class_num, self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

        ##INIT LOGGER
        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL) ## set log config
        console = logging.StreamHandler() # console out
        console.setLevel(CONSOLE_LEVEL) # set log level
        logging.getLogger().addHandler(console)

    def choose_action(self, history, epsilon=None):
        if epsilon is not None and np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).to(self.device)
                action = self.model(state) if self.device == 'cpu' else self.model(state).cpu()
                return int(action.max(0).indices.numpy())

    def get_epsilon(self, t):
        epsilon =  self.eps_end + max(0, (self.ep - self.eps_end)*(self.eps_endt - max(0, t - self.learn_start)) /self.eps_endt )
        return epsilon

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        next_state = torch.stack(batch.next_state)
        reward = torch.stack(batch.reward)
        done = torch.stack(batch.done)
        life = torch.stack(batch.life)
        with torch.no_grad():
            next_state_action_values = self.model(next_state)
        next_state_value = torch.max(next_state_action_values, dim=1).values.view(-1, 1)
        reward = reward.view(-1, 1)
        target_state_value = torch.stack([reward + (self.discount * next_state_value), reward], dim=1).squeeze().gather(1,
                                                                                                                     done)

        self.optimizer.zero_grad()
        state_action_values = self.model(state).gather(1, action)
        loss = F.mse_loss(state_action_values, target_state_value)
        loss.backward()
        self.optimizer.step()

    def run(self):
        progress_bar = tqdm(range(self.max_steps))
        state = self.env.reset()
        history = historyDataset(self.history_size, state, self.device)
        done = False

        ##Report
        scores = deque(maxlen=10)
        score = 0
        episode = 0
        max_score = 0

        for step in progress_bar:

            ##Terminal
            if done:
                state = self.env.reset()
                history = historyDataset(self.history_size, state, self.device)
                scores.append(score)
                if score > max_score:
                    max_score = score
                score = 0
                episode += 1


            action = self.choose_action(history, self.get_epsilon(step))
            next_state, reward, done, life = self.env.step(action)
            state = history.get_state()
            history.push(next_state)
            next_state = history.get_state()
            life = life['ale.lives']
            self.memory.push(state, action, reward, next_state, done, life)
            if step > self.learn_start and step % self.update_freq == 0:
                self.replay(self.batch_size)

            score += reward

            if step % self.eval_freq == 0:
                mean_score = np.mean(scores)
                progress_bar.set_postfix_str(
                    '[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode, mean_score,
                                                                                       max_score,
                                                                                       self.get_epsilon(episode)))
                logging.debug('[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode, mean_score,
                                                                                                 max_score,
                                                                                                 self.get_epsilon(
                                                                                                     episode)))

if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and config.device in ["gpu",'cuda'] else "cpu")
    config.device = device
    agent = DQNSolver(config)
    agent.run()




