import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
import math
import random
from collections import namedtuple
from PIL import Image, ImageDraw
import os


class cnnDQN(nn.Module):

    def __init__(self, resize_unit, history_size, outputs, device):
        h, w = resize_unit
        super(cnnDQN, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(history_size, 16, kernel_size=5, stride=2).to(device)
        self.bn1 = nn.BatchNorm2d(16).to(device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2).to(device)
        self.bn2 = nn.BatchNorm2d(32).to(device)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2).to(device)
        self.bn3 = nn.BatchNorm2d(32).to(device)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


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


class DQNSolver():
    def __init__(self, n_episodes=10000, max_env_steps=None, gamma=1.0,
                 epsilon=1.0, epsilon_min=0.01, class_num=4, epsilon_log_decay=0.995,
                 alpha=0.01, alpha_decay=0.01, batch_size=4, monitor=True, quiet=False, device='cpu',
                 pretrained=None, rendering=False, history_size=2, resize_unit=(161, 144)):
        self.device = device
        self.memory = memoryDataset(maxlen=100000, device=device)
        self.env = gym.make('BreakoutDeterministic-v4')
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.quiet = quiet
        self.class_num = class_num
        self.model = cnnDQN(resize_unit, history_size, class_num, self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=alpha, weight_decay=alpha_decay)
        self.resize_unit = resize_unit
        self.history_size = history_size

        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        save_folder = os.path.join(os.getcwd(), 'model')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_path = os.path.join(save_folder, 'model.pkl')

    def choose_action(self, state, epsilon=None):
        if epsilon is not None and np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                action = self.model(state) if self.device == 'cpu' else self.model(state).cpu()
                return int(action.max(dim=1).indices.numpy())

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

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
        target_state_value = torch.stack([reward + (self.gamma * next_state_value), reward], dim=1).squeeze().gather(1,
                                                                                                                     done)

        self.optimizer.zero_grad()
        state_action_values = self.model(state).gather(1, action)
        loss = F.mse_loss(state_action_values, target_state_value)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def init_history(self, state):
        temp = []
        for _ in range(self.history_size):
            temp.append(state)
        return torch.stack(temp)

    def push_history(self, history, state):
        return torch.cat([history, state.unsqueeze(0)])[1:]

    def convert_channel(self, img):
        # input type : |img| = (Height, Width, channel)

        # remove useless item
        img = img[32:193, 8:152]

        # conver channel(3) -> channel(1)
        img = np.any(img, axis=2)
        # |img| = (Height, Width)  boolean
        return torch.tensor(img, dtype=torch.float)

    def run(self):
        scores = deque(maxlen=100)
        lives = deque(maxlen=100)
        for episode in range(self.n_episodes):
            score = 0
            state = self.env.reset()
            state = self.convert_channel(state)
            history = self.init_history(state)
            done = False
            i = 0
            while not done:
                action = self.choose_action(history, self.get_epsilon(episode))
                next_state, reward, done, life = self.env.step(action)
                next_state = self.convert_channel(next_state)
                life = life['ale.lives']
                next_history = self.push_history(history, next_state)
                self.memory.push(history, action, reward, next_history, done, life)
                history = next_history
                score += reward

            scores.append(score)
            lives.append(life)
            mean_score = np.mean(scores)  ##최근 100개가 버티는 시간의 Mean이 조건을 만족하면 멈춤..
            mean_life = np.mean(lives)
            if life > 0 and done:
                if not self.quiet: print('{} episodes. Solved after {} trials '.format(episode, episode - 100))
                SAVE_PATH = '/model/model.pkl'
                self.save_model()

                return episode - 100
            if episode % 100 == 0 and not self.quiet:
                print('[Episode {}] - last 100 episodes score : {}, life : {}, epsilon : {}'.format(episode, mean_score,
                                                                                                    mean_life,
                                                                                                    self.epsilon))

            self.replay(self.batch_size)

        if not self.quiet: print('Did not solve after {} episodes'.format(episode))
        return episode

    def save_model(self):
        param_groups = {}
        param_groups['model_state_dict'] = self.model.state_dict()
        param_groups['optimizer_state_dict'] = self.optimizer.state_dict()
        # param_groups['class_num'] = self.class_num
        param_groups['gamma'] = self.gamma
        param_groups['epsilon'] = self.epsilon
        param_groups['alpha'] = self.alpha
        # param_groups['alpha_decay'] = self.alpha_decay
        param_groups['batch_size'] = self.batch_size
        torch.save(param_groups, self.save_path)

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def render_policy_net(self):
        self.load_model()
        # self.env = gym.wrappers.Monitor(self.env, os.path.join(os.getcwd(), 'model','cartpole-1'), force=True)
        state = self.env.reset()
        done = False
        frames = []
        raw_frames = []
        i = 0
        while not done:
            img = self.env.render(mode='rgb_array')
            raw_frames.append(img)
            img = Image.fromarray(img)
            frames.append(img)
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            i = i + 1
        print(i)
        self.env.close()
        frames[0].save('Breakout_result.gif', format='GIF', append_images=frames[1:], save_all=True, duration=0.0001)
        print("save picture -- Breakout_result.gif")
        return frames, raw_frames

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNSolver(device=device)
    agent.run()
    #frames, raw_frames = agent.render_policy_net()

