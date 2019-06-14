import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import logging
from model import DQN, DQN_CNN
from properties import build_parser, CONSOLE_LEVEL, LOG_FILE, LOGFILE_LEVEL
from repository import historyDataset, memoryDataset



class DQNSolver():

    def __init__(self, config):
        self.device = config.device
        self.env = gym.make(config.env)
        self.memory_size = config.memory_size
        self.update_freq = config.update_freq
        self.learn_start = config.learn_start
        self.history_size = config.history_size


        self.batch_size = config.batch_size
        self.ep = config.ep
        self.eps_end = config.eps_end
        self.eps_endt = config.eps_endt
        self.lr = config.lr
        self.discount = config.discount

        self.agent_type = config.agent_type
        self.max_steps = config.max_steps
        self.eval_freq = config.eval_freq
        self.eval_steps = config.eval_steps
        self.target_update = config.target_update


        ##Breakout Setting
        self.resize_unit = (84, 84)
        self.class_num = 4
        #self.resize_unit = (161, 144)

        ##INIT SETTING
        self.memory = memoryDataset(maxlen=config.memory_size, device=config.device)

        self.hidden_size = 256
        self.policy_model = DQN_CNN(self.history_size, self.hidden_size, self.class_num, self.device).to(self.device)
        self.target_model = DQN_CNN(self.history_size, self.hidden_size, self.class_num, self.device).to(self.device)
        #self.policy_model = DQN(self.history_size, self.resize_unit[0], self.resize_unit[1], self.class_num).to(self.device)
        #self.target_model = DQN(self.history_size, self.resize_unit[0], self.resize_unit[1], self.class_num).to(self.device)

        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=self.lr)

        ##INIT LOGGER
        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL) ## set log config
        console = logging.StreamHandler() # console out
        console.setLevel(CONSOLE_LEVEL) # set log level
        logging.getLogger().addHandler(console)

        ##save options
        save_folder = os.path.join(os.getcwd(), 'model')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_path = os.path.join(save_folder, 'model.pkl')

        self.score_memory = []
        self.score_save_path = os.path.join(save_folder, 'score')



    def choose_action(self, history, epsilon=None):
        if epsilon is not None and np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to(self.device)
                action = self.target_model(state) if self.device == 'cpu' else self.target_model(state).cpu()
                return int(action.max(1).indices.numpy())

    def get_epsilon(self, t):
        epsilon =  self.eps_end + max(0, (self.ep - self.eps_end)*(self.eps_endt - max(0, t - self.learn_start)) /self.eps_endt )
        return epsilon

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)

        ##device 변경 cpu to gpu
        state = torch.stack(batch.state).to(self.device)
        action = torch.stack(batch.action).to(self.device)
        next_state = torch.stack(batch.next_state).to(self.device)
        reward = torch.stack(batch.reward)

        ## reward rescale 0 ~ 1
        reward = reward.type(torch.bool).type(torch.float).to(self.device)

        done = torch.stack(batch.done).to(self.device)
        life = torch.stack(batch.life)
        terminal = torch.stack(batch.terminal).to(self.device)

        with torch.no_grad():
            next_state_action_values = self.policy_model(next_state)
        next_state_value = torch.max(next_state_action_values, dim=1).values.view(-1, 1)
        reward = reward.view(-1, 1)
        target_state_value = torch.stack([reward + (self.discount * next_state_value), reward], dim=1).squeeze().gather(1, terminal)

        self.optimizer.zero_grad()
        state_action_values = self.policy_model(state).gather(1, action)
        loss = F.mse_loss(state_action_values, target_state_value)
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        param_groups = {}
        param_groups['model_state_dict'] = self.policy_model.state_dict()
        torch.save(param_groups, self.save_path)

    def run(self):
        progress_bar = tqdm(range(self.max_steps))
        state = self.env.reset()
        history = historyDataset(self.history_size, state)
        done = False

        ##Report
        scores = deque(maxlen=10)
        score = 0
        episode = 0
        max_score = 0
        last_life = 0

        for step in progress_bar:

            ## model update
            if step > self.learn_start and step % self.target_update == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())

            ## game is over
            if done:
                state = self.env.reset()
                history = historyDataset(self.history_size, state)
                scores.append(score)
                if score > max_score:
                    max_score = score
                    self.save_model()
                score = 0
                last_life = 0
                episode += 1

                if episode % 100 == 0:
                    self.score_memory.append(np.mean(scores))
                    if episode % 100000 == 0:
                        np.save(self.score_save_path, self.score_memory)

            action = self.choose_action(history, self.get_epsilon(step))
            next_state, reward, done, life = self.env.step(action)
            state = history.get_state()
            history.push(next_state)
            next_state = history.get_state()
            life = life['ale.lives']

            ## Terminal options
            if life < last_life:
                terminal = True
            else :
                terminal = False
            last_life = life


            self.memory.push(state, action, reward, next_state, done, life, terminal)
            if step > self.learn_start and step % self.update_freq == 0:
                self.replay(self.batch_size)

            score += 1 if reward > 0 else 0

            if step > self.eval_steps and step % self.eval_freq == 0:
                mean_score = np.mean(scores)
                progress_bar.set_postfix_str(
                    '[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode, mean_score,
                                                                                       max_score,
                                                                                       self.get_epsilon(step)))
                logging.debug('[Episode %s] - score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode, mean_score,
                                                                                                 max_score,
                                                                                                 self.get_epsilon(
                                                                                                     step)))

if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and config.device in ["gpu",'cuda'] else "cpu")
    config.device = device
    agent = DQNSolver(config)
    agent.run()