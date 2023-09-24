# from maze_env import Maze
from RL_brain import DeepQNetwork
import numpy as np
import time


class My_environment(object):
    def __init__(self, action_space, observation_space, action_space_high, action_space_low):
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.n_actions = 6
        self.n_features = self.observation_space

    def env_random_reset(self):
        a_random_reset = np.empty(self.action_space, dtype='float32')
        s_random_reset = 0
        for i in range(self.action_space):
            a_random_reset[i] = np.clip(np.random.normal(0, 0.1*(self.action_space_high[i] - self.action_space_low[i])), self.action_space_low[i], self.action_space_high[i])
            # s_random_reset += (a_random_reset[i] - i - 1)**2
        a_random_reset = np.round(a_random_reset, 3)
        s_random_reset = a_random_reset
        observation = s_random_reset
        return observation

    def step(self, state, action):
        s_ = state
        y = 0
        for i in range(len(s_)):
            y += (s_[i] - i - 1)**2
        if action == 0:
            s_[0] += 1e-3
        elif action == 1:
            s_[0] -= 1e-3
        elif action == 2:
            s_[1] += 1e-3
        elif action == 3:
            s_[1] -= 1e-3
        elif action == 4:
            s_[2] += 1e-3
        elif action == 5:
            s_[2] -= 1e-3

        s_ = np.round(s_, 3)
        y_ = 0
        for k in range(len(s_)):
            y_ += (s_[k] - k - 1)**2

        if y_ == 0:
            reward = 100
            done = True
        else:
            reward = y - y_
            done = False
        # elif y - y_ < 0:
        #     reward = -1
        #     done = False


        return s_, reward, done

def play():
    step = 0
    for episode in range(300):
        # t0 = time.time()
        # initial observation
        # observation = env.reset()
        observation = env.env_random_reset()
        i = 0

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)  #根据state随机抽取action的动作是由神经网络完成的
            t0 = time.time()
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(observation, action)
            # print("reward:{} \n episode:{}".format(reward, episode))

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            t1 = time.time()
            print("episode:{}  ||  reward:{}  ||  state:{}  ||  time:{}ms".format(episode, reward, observation, 1000*(t1 - t0)))
            # break while loop when end of this episode
            i += 1

            if done:
                # t1 = time.time()
                # print("episode:{}  ||  reward:{}  ||  state:{}  ||  time:{}".format(episode, reward, observation, (t1-t0)))
                break
            step += 1

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    # state_size = 3
    # env = Maze()
    env = My_environment(
        action_space=3,
        observation_space=3,
        action_space_high=[100, 100, 100],
        action_space_low=[-100, -100, -100],
    )
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    # env.after(100, run_maze)
    # env.mainloop()
    play()
    RL.plot_cost()