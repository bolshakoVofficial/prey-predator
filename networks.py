import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam


class CNN:
    def __init__(self, env, replay_memory_size, min_replays, batch_size, update_target_every, gamma, model_name=None,
                 lr=0.001):
        self.input_size = env.pixel_state_shape
        self.n_actions = env.preys[0].n_actions
        self.lr = lr
        self.gamma = gamma
        self.min_replays = min_replays
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        # queue, that provide append and pop to store replays
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

        # evaluation network model (trains every step)
        self.model_name = model_name
        self.model = self.create_model()

        # target model (predicts every step, updates every n steps)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        if self.model_name:
            model = load_model(self.model_name)
            print(f"Loaded model: {self.model_name}")
        else:
            model = Sequential()
            # 1st layer
            model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=self.input_size))
            model.add(Activation("relu"))
            model.add(Dropout(0.5))

            # 2nd layer
            model.add(Conv2D(32, (4, 4), strides=(2, 2)))
            model.add(Activation("relu"))
            model.add(Dropout(0.5))

            # 3rd layer
            model.add(Flatten())
            model.add(Dense(256))
            model.add(Dropout(0.5))

            # output layer
            model.add(Dense(self.n_actions, activation="linear"))
            model.compile(optimizer=Adam(self.lr), loss="mse", metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        # step of env is transition - (current_state, action, reward, new_state, terminated)
        self.replay_memory.append(transition)

    def get_q_values(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replays:
            # not enough replays for training
            return

        # train on random replays from memory
        batch = random.sample(self.replay_memory, self.batch_size)

        # from eval net
        current_states = np.array([transition[0] for transition in batch]) / 255
        current_qs_list = self.model.predict(current_states)

        # from target net
        new_current_states = np.array([transition[3] for transition in batch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        # observations and actions (a.k.a. features and labels)
        X = []
        y = []

        for index, (current_state, action, reward, new_state, terminated) in enumerate(batch):
            if not terminated:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # update q
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.batch_size,
                       verbose=0, shuffle=False, callbacks=None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            # updating weights of target net
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class MLP:
    def __init__(self, env, replay_memory_size, min_replays, batch_size, update_target_every, gamma, model_name=None,
                 lr=0.001, input_size=None, n_actions=None):
        self.input_size = input_size
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.min_replays = min_replays
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        # queue, that provide append and pop to store replays
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

        # evaluation network model (trains every step)
        self.model_name = model_name
        self.model = self.create_model()

        # target model (predicts every step, updates every n steps)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        if self.model_name:
            model = load_model(self.model_name)
            print(f"Loaded model: {self.model_name}")
        else:
            model = Sequential()
            model.add(Dense(64, input_shape=self.input_size))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            model.add(Dense(32))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            model.add(Dense(self.n_actions, activation="linear"))
            model.compile(optimizer=Adam(lr=self.lr), loss="mse", metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        # step of env is transaction - (current_state, action, reward, new_state, terminated)
        self.replay_memory.append(transition)

    def get_q_values(self, state):
        # from eval net
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replays:
            # not enough replays for learning
            return

        # train on random replays from memory
        batch = random.sample(self.replay_memory, self.batch_size)

        # from eval net
        current_states = np.array([transition[0] for transition in batch]) / 255
        current_qs_list = self.model.predict(current_states)

        # from target net
        new_current_states = np.array([transition[3] for transition in batch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        # observations and actions (a.k.a. features and labels)
        X = []
        y = []

        for index, (current_state, action, reward, new_state, terminated) in enumerate(batch):
            if not terminated:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # update q
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.batch_size,
                       verbose=0, shuffle=False, callbacks=None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            # updating weights of target net
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
