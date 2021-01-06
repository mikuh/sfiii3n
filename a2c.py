import tensorflow as tf
import numpy as np
import gym
import random
from sfiii3n_env import Sfiii3nEnv


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


class A2C(tf.keras.Model):

    def __init__(self, embedding_net, policy_net, value_net):
        super(A2C, self).__init__()

        self._embedding_layer = embedding_net
        self._policy_layer = policy_net
        self._value_layer = value_net

    def call(self, state):
        embedding = self._embedding_layer(state)
        policy = self._policy_layer(embedding)
        value = self._value_layer(embedding)
        return policy, value


class EmbeddingNet(tf.keras.layers.Layer):

    def __init__(self, env):
        super().__init__()
        self.layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                                             input_shape=env.state_shape)
        self.layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.layer3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.layer4 = tf.keras.layers.Flatten()
        self.layer5 = tf.keras.layers.Dense(256, activation='relu')

    def call(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output


class PolicyNet(tf.keras.layers.Layer):
    def __init__(self, env):
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(env.action_space.n, activation='softmax')
        super().__init__()

    def call(self, inputs):
        inputs = self.layer1(inputs)
        return self.logits(inputs)


class ValueNet(tf.keras.layers.Layer):

    def __init__(self):
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)
        super().__init__()

    def call(self, inputs):
        inputs = self.layer1(inputs)
        return self.value(inputs)


class Agent(object):

    def __init__(self, model, env):

        self.env = env
        self.action_size = self.env.action_space.n
        self.a2c = model
        self.n_step = 1
        self.gamma = 0.99
        self.rollout = 4096
        self.batch_size = 256
        self.lr = 0.001
        self.epsilon = 0.5
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        self.temperature = 1
        self.episode = 0
        self.score = 0
        self.best_score = 0

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.a2c(state)
        policy = np.array(policy)[0]
        self.epsilon = 1 / (self.episode * 0.1 + 2)
        if random.random() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            # where_are_nan = np.isnan(policy)
            # policy[where_are_nan] = 1e-8
            # policy = softmax(policy/temperature)
            action = np.random.choice(self.action_size, p=policy)
        return action

    def collect_replay_buffer(self, state):
        state_list, next_state_list, reward_list, done_list, action_list = [], [], [], [], []

        for _ in range(self.rollout):
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.score += reward


            state_list.append(state)
            next_state_list.append(next_state)
            reward_list.append(reward)
            done_list.append(done)
            action_list.append(action)

            state = next_state

            if done > 0:
                print("Episode %s, Score: %s" % (self.episode, self.score))
                state = self.env.reset(done)
                self.episode += 1
                if self.episode % 100 == 0:
                    # self.best_score = self.score
                    self.a2c.save("my_model/")
                self.score = 0

        if self.n_step > 1:
            states, next_states, rewards, dones, actions = [], [], [], [], []
            for i in range(len(state_list) - self.n_step + 1):
                state = state_list[i]
                next_state = next_state_list[i + self.n_step - 1]
                reward = 0
                for index, x in enumerate(reward_list[i: i + self.n_step]):
                    reward += self.gamma ** index * x
                done = done_list[i]
                action = action_list[i]

                states.append(state)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                actions.append(action)
            return states, next_states, rewards, dones, actions

        return state_list, next_state_list, reward_list, done_list, action_list

    def learn(self, max_episode):
        init_state = self.env.init_state
        while self.episode < max_episode:
            _state, _next_state, _reward, _done, _action = self.collect_replay_buffer(init_state)
            for _ in range(20):
                sample_range = np.arange(self.rollout - self.n_step + 1)
                np.random.shuffle(sample_range)
                sample_idx = sample_range[:self.batch_size]

                state = [_state[i] for i in sample_idx]
                next_state = [_next_state[i] for i in sample_idx]
                reward = [_reward[i] for i in sample_idx]
                done = [_done[i] for i in sample_idx]
                action = [_action[i] for i in sample_idx]

                a2c_variable = self.a2c.trainable_variables
                with tf.GradientTape() as tape:
                    tape.watch(a2c_variable)

                    _, current_value = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
                    _, next_value = self.a2c(tf.convert_to_tensor(next_state, dtype=tf.float32))
                    current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)


                    target = tf.stop_gradient(
                        self.gamma * (1 - tf.convert_to_tensor(done, dtype=tf.float32)) * next_value + tf.convert_to_tensor(
                            reward, dtype=tf.float32))
                    value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

                    policy, _ = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
                    # policy = tf.nn.softmax(policy, axis=None, name=None)
                    entropy = tf.reduce_mean(- policy * tf.math.log(policy + 1e-8)) * 0.1
                    action = tf.convert_to_tensor(action, dtype=tf.int32)
                    onehot_action = tf.one_hot(action, self.action_size)
                    action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
                    adv = tf.stop_gradient(target - current_value)
                    pi_loss = -tf.reduce_mean(tf.math.log(action_policy + 1e-8) * adv) - entropy

                    total_loss = pi_loss + value_loss

                grads = tape.gradient(total_loss, a2c_variable)
                self.opt.apply_gradients(zip(grads, a2c_variable))

        self.a2c.save("my_model/")

    def play(self):
        obs = self.env.reset()
        score = 0
        while True:
            action = self.get_action(obs)
            obs, rewards, dones, info = self.env.step(action)
            score += rewards
            self.env.render()
            if dones > 0:
                print("得分:", score)
                obs = self.env.reset()
                score = 0


if __name__ == '__main__':
    env = Sfiii3nEnv()

    a2c = A2C(embedding_net=EmbeddingNet(env),
              policy_net=PolicyNet(env),
              value_net=ValueNet())
    # a2c = tf.keras.models.load_model("best_model/")
    agent = Agent(model=a2c, env=env)

    agent.learn(1000000)

    # play
    # a2c = tf.keras.models.load_model("best_model/")
    # agent = Agent(model=a2c, gym_env_name="SpaceInvaders-ram-v0")
    # agent.play()
