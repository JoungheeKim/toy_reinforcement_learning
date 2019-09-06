# toy_reinforcement_learning
This is a project to test Atari toy example with reinforcement learning algorithm. In this project, DQN is used to reach the score that [the paper](https://arxiv.org/pdf/1312.5602.pdf) reported. You can check good [lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&t=7s) and [papers](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) about RL

### Environment
Thanks to OpenAI, They freely give us wonderful example environment which is called GYM. You can check [web site](https://gym.openai.com/) to see what they have. In this project, I use two environment ( breakout v4, cartpole )

### Result Report
1. cartpole
  Survived 195 steps. you can check the easy [python code](https://github.com/JoungheeKim/toy_reinforcement_learning/blob/master/cartpole/cartpole_tutorial.ipynb)
  ![alt text](https://github.com/JoungheeKim/toy_reinforcement_learning/blob/master/cartpole/CartPole_result.gif)
2. breakout v2
  In the paper, DeepMind reports an evaluation score of 317 for Breakout. But I only got maximum 71. The reason is the history size that i put as 2, but i should put 4 instead like DeepMind does.
  ![alt text](https://github.com/JoungheeKim/toy_reinforcement_learning/blob/master/breakout_v2/evaluation_score(history%202).png)
