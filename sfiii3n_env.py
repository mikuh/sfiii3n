import gym
from gym import spaces
from MAMEToolkit.sf_environment import Environment
import tensorflow as tf
import numpy as np

action_space = []
for move_action in range(0, 8 + 1):
    for attack_action in range(0, 9 + 1):
        action_space.append((move_action, attack_action))

N_DISCRETE_ACTIONS = len(action_space)

penalty = 1 / 60 * 3


class Sfiii3nEnv(gym.Env):
    """CSfiii3n Environment that follows gym interface"""

    def __init__(self, rom_path="/home/miku/roms", env_id=1, difficulty=3, frame_ratio=3, frames_per_step=3, render=False):
        super(Sfiii3nEnv, self).__init__()
        self.env = Environment(f"env{env_id}", roms_path=rom_path, difficulty=difficulty, frame_ratio=frame_ratio,
                               frames_per_step=frames_per_step)
        self.init_state = self.env.start()
        self.frames_per_step = frames_per_step
        if self.frames_per_step > 1:
            self.init_state = self.init_state[-1]
        # while len(self.init_state) < 3:
        #     self.init_state.append(observation[-1])
        # self.init_state = tf.concat(self.init_state, axis=-1)
        self.state_shape = self.init_state.shape
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.observation_space = spaces.Box(low=0, high=255, shape=self.state_shape, dtype=np.uint8)

    def step(self, action):
        move_action, attack_action = action_space[action]
        observation, reward, round_done, stage_done, game_done = self.env.step(move_action, attack_action)
        done = 0
        if game_done:
            done = 1
        elif stage_done:
            done = 2
        elif round_done:
            done = 3
        reward = reward['P1'] - penalty
        # if done > 0:
        #     if self.env.expected_wins["P1"] == 2:
        #         reward += 1000
        # while len(observation) < 3:
        #     observation.append(observation[-1])
        # observation = tf.concat(observation, axis=-1)
        if self.frames_per_step > 1:
            observation = observation[-1]
        return observation, reward, done, ""

    def reset(self, done):
        if done == 1:
            observation = self.env.new_game()
        elif done == 2:
            observation = self.env.next_stage()
        elif done == 3:
            observation = self.env.next_round()
        # while len(observation) < 3:
        #     observation.append(observation[-1])
        # observation = tf.concat(observation, axis=-1)
        if self.frames_per_step > 1:
            observation = observation[-1]
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        del self.env


if __name__ == '__main__':
    env = Sfiii3nEnv()

    while True:
        action = np.random.choice(list(range(env.action_space.n)), 1)[0]
        observation, reward, done, _ = env.step(action)
        # time.sleep(1)
        if done > 0:
            env.reset(done)
