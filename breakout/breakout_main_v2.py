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
from tqdm import tqdm
import logging

##LOGGING PROPERTY
LOG_FILE = 'logfile'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG


class myDQN(nn.Module):

    def __init__(self, init_dim, layers, class_num, device):
        super(myDQN, self).__init__()
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


class myHistory(object):
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
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=1.0, layers=[100], run_start=1000,
                 epsilon=1.0, epsilon_min=0.01, class_num=4, epsilon_log_decay=0.995, memory_size = 100000,
                 alpha=0.000125, alpha_decay=0.01, batch_size=32, freq_step=4, quiet=False, device='cpu',
                 pretrained=None, rendering=False, history_size=2, resize_unit=(161, 144)):
        self.device = device
        self.memory = memoryDataset(maxlen=memory_size, device=device)
        self.memory_size = memory_size
        self.env = gym.make('BreakoutDeterministic-v4')
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.run_start = run_start
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.freq_step = freq_step
        self.quiet = quiet
        self.layers = layers
        self.class_num = class_num
        self.init_dim = 5185 + (2 * history_size)
        self.model = myDQN(self.init_dim, layers, class_num, self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=alpha)
        self.resize_unit = resize_unit
        self.history_size = history_size

        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        save_folder = os.path.join(os.getcwd(), 'model')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_path = os.path.join(save_folder, 'model.pkl')

        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL) #logging의 config 변경
        console = logging.StreamHandler() #logging을 콘솔화면에 출력
        console.setLevel(CONSOLE_LEVEL) # log level 설정
        logging.getLogger().addHandler(console) #logger 인스턴스에 콘솔창의 결과를 핸들러에 추가한다.

    def choose_action(self, history, epsilon=None):
        if epsilon is not None and np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).to(self.device)
                action = self.model(state) if self.device == 'cpu' else self.model(state).cpu()
                return int(action.max(0).indices.numpy())

    def get_epsilon(self, t):
        epsilon =  self.epsilon_min + max(0, (self.epsilon - self.epsilon_min)*(self.memory_size - max(0, t - self.run_start)) /self.memory_size )
        #epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

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
        target_state_value = torch.stack([reward + (self.gamma * next_state_value), reward], dim=1).squeeze().gather(1,
                                                                                                                     done)

        self.optimizer.zero_grad()
        state_action_values = self.model(state).gather(1, action)
        loss = F.mse_loss(state_action_values, target_state_value)
        loss.backward()
        self.optimizer.step()



    def run(self):
        scores = deque(maxlen=10)
        max_score = 0.0
        i = 0
        progress_bar = tqdm(range(self.n_episodes))
        for episode in progress_bar:
            score = 0
            state = self.env.reset()
            history = myHistory(self.history_size, state, self.device)
            done = False
            while not done:
                action = self.choose_action(history, self.get_epsilon(i))
                next_state, reward, done, life = self.env.step(action)
                state = history.get_state()
                history.push(next_state)
                next_state = history.get_state()
                life = life['ale.lives']
                self.memory.push(state, action, reward, next_state, done, life)
                score += reward
                i = i+1
                if i>self.run_start and i%self.freq_step == 0:
                    self.replay(self.batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            scores.append(score)
            mean_score = np.mean(scores)  ##최근 100개가 버티는 시간의 Mean이 조건을 만족하면 멈춤..
            if max_score < mean_score:
                max_score = mean_score

            if life > 0 and done:
                if not self.quiet:
                    progress_bar.set_postfix_str('{} episodes. Solved after {} trials '.format(episode, episode - 10))
                    logging.debug('{} episodes. Solved after {} trials '.format(episode, episode - 10))
                SAVE_PATH = '/model/model.pkl'
                self.save_model()

                return episode - 100
            if episode % 10 == 0 and not self.quiet:
                progress_bar.set_postfix_str('[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' %(episode, mean_score,
                                                                                                    max_score,
                                                                                                    self.get_epsilon(episode)))
                logging.debug('[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' %(episode, mean_score,
                                                                                                    max_score,
                                                                                                    self.get_epsilon(episode)))

        if not self.quiet:
            progress_bar.set_postfix_str('Did not solve after {} episodes'.format(episode))
            logging.debug('Did not solve after {} episodes'.format(episode))
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
    # device = 'cpu'
    print(device)
    layers = [250]
    class_num = 4
    agent = DQNSolver(layers=layers, class_num=class_num, device=device)
    agent.run()
    # frames, raw_frames = agent.render_policy_net()

