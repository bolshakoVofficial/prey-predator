import pygame as pg
import numpy as np
import cv2
import networks
import os
import time
import pickle
import matplotlib.pyplot as plt
from env import Env
from tqdm import tqdm

np.set_printoptions(suppress=True)

if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('models/predator'):
    os.makedirs('models/predator')
if not os.path.isdir('models/prey'):
    os.makedirs('models/prey')
if not os.path.isdir('stats'):
    os.makedirs('stats')

n_steps = 200
raw_map_view = False
show_f_map = False
show_env = True
f_map_size_multiplier = 5
predators_number = 2
preys_number = 1

EPS_WITH_EPSILON = 2000
EPS_WITH_ZERO_EPSILON = EPS_WITH_EPSILON // 10
EPISODES = EPS_WITH_EPSILON + EPS_WITH_ZERO_EPSILON
epsilon = 1
EPSILON_DECAY_RATE = epsilon / EPS_WITH_EPSILON
MIN_EPSILON = 0
ALPHA = 0.001  # here, for NN optimizer lr
GAMMA = 0.99

REPLAY_MEMORY_SIZE = 3200  # n last steps of env for training
MIN_REPLAY_MEMORY_SIZE = 800  # minimum n of steps in a memory to start training
BATCH_SIZE = 64  # samples for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
AGGREGATE_STATS_EVERY = 20  # episodes

# prey_model_name = "models/prey/1pd1p_cnn16x32x256_id0_0.48avg_40steps.h5"
prey_model_name = None
predator_model_name = None

# stats for matplotlib
save_stats_every = 10
stat_epsilon_values = []
stat_total_steps = []
stat_predators_ep_reward = []
stat_prey_ep_reward = []
stat_mean_total_steps = []
stat_mean_predators_ep_reward = []
stat_mean_prey_ep_reward = []
nn_save_predator_ep_rewards = []
nn_save_prey_ep_rewards = []
nn_save_predator_new_average_reward = 0
nn_save_prey_new_average_reward = 0
nn_save_prey_new_average_steps = 40

if __name__ == '__main__':
    env = Env(raw_map_view=raw_map_view, draw_env_window=show_env, predators_number=predators_number,
              preys_number=preys_number, max_env_ticks=n_steps)
    MODEL_NAME = f"{env.predators_number}pd{env.preys_number}p_cnn16x32x256"

    for prey in env.preys:
        # prey.nn = networks.MLP(env, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, BATCH_SIZE, UPDATE_TARGET_EVERY,
        #                        GAMMA, lr=ALPHA, model_name=prey_model_name, input_size=env.prey_state_shape,
        #                        n_actions=env.prey_n_actions)
        prey.nn = networks.CNN(env, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, BATCH_SIZE, UPDATE_TARGET_EVERY, GAMMA,
                               lr=ALPHA, model_name=prey_model_name)

    for predator in env.predators:
        predator.nn = networks.MLP(env, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, BATCH_SIZE, UPDATE_TARGET_EVERY,
                                   GAMMA,
                                   lr=ALPHA, model_name=prey_model_name, input_size=env.predator_state_shape,
                                   n_actions=env.predator_n_actions)

    pg.init()
    screen = pg.display.set_mode(env.screen_size)
    pg.display.set_caption("Preys and Predators")
    clock = pg.time.Clock()

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
        env.reset()

        terminated = False
        predator_episode_reward = 0
        prey_episode_reward = 0

        timings = np.array([0.] * 2)
        prey_train1, prey_train2 = 0, 0
        predator_train1, predator_train2 = 0, 0

        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY_RATE
        else:
            epsilon = MIN_EPSILON

        while not terminated:
            preys = env.preys
            predators = env.predators

            predator_actions = []
            prey_actions = []
            prey_rewards = []
            predator_state, prey_state = env.get_state()
            pixel_state = env.get_pixel_history()
            # pixel_state = env.get_pixel_state()

            if show_f_map:
                f_map = env.get_pixel_state()
                # f_map = env.get_pixel_history()

                # f_map_size_multiplier times bigger than CNN gets
                resized_image = cv2.resize(f_map, dsize=None, fx=f_map_size_multiplier,
                                           fy=f_map_size_multiplier)
                cv2.imshow(f"Feature map ({f_map_size_multiplier} times bigger)", resized_image)
                cv2.waitKey(1)

            # for predator in predators:
            #     avail_actions = predator.available_actions(env.env_map)
            #     avail_actions_index = np.nonzero(avail_actions)[0]
            #
            #     if np.random.random() > epsilon:
            #         q_values = predator.nn.get_q_values(predator_state)
            #         avail_qs = [q_values[i] if i in avail_actions_index else -1000 for i in range(len(q_values))]
            #         action = np.argmax(avail_qs)
            #     else:
            #         action = np.random.choice(avail_actions_index)
            #
            #     predator_actions.append(action)

            for prey in preys:
                avail_actions = prey.available_actions(env.env_map)
                avail_actions_index = np.nonzero(avail_actions)[0]

                if np.random.random() > epsilon:
                    q_values = prey.nn.get_q_values(pixel_state)
                    avail_qs = [q_values[i] if i in avail_actions_index else -1000 for i in range(len(q_values))]
                    action = np.argmax(avail_qs)
                else:
                    action = np.random.choice(avail_actions_index)

                prey_actions.append(action)

            predators_reward, prey_rewards, terminated = env.step(predator_actions, prey_actions)

            predator_episode_reward += predators_reward
            prey_episode_reward += sum(prey_rewards.values())

            new_predator_state, new_prey_state = env.get_state()
            new_pixel_state = env.get_pixel_history()
            # new_pixel_state = env.get_pixel_state()

            for i, prey in enumerate(preys):
                if not prey.is_alive() and (prey.how_long_not_alive() > 1):
                    pass
                else:
                    prey.nn.update_replay_memory(
                        (pixel_state, prey_actions[i], prey_rewards[prey.id], new_pixel_state, terminated))

                    prey_train1 = time.time()
                    prey.nn.train(terminated)
                    prey_train2 = time.time()

            # for i, predator in enumerate(predators):
            #     predator.nn.update_replay_memory((predator_state,
            #                                       predator_actions[i],
            #                                       predators_reward,
            #                                       new_predator_state,
            #                                       terminated))
            #
            #     predator_train1 = time.time()
            #     predator.nn.train(terminated)
            #     predator_train2 = time.time()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pass
            if env.draw_env_window:
                env.draw_env(screen)

            pg.display.update()
            env.increment_env_tick()

            timings += np.array([prey_train2 - prey_train1, predator_train2 - predator_train1])

        timings = np.around(timings, 3)

        nn_save_predator_ep_rewards.append(predator_episode_reward)
        nn_save_prey_ep_rewards.append(prey_episode_reward)

        # for matplotlib
        stat_predators_ep_reward.append(predator_episode_reward)
        stat_prey_ep_reward.append(prey_episode_reward)
        stat_total_steps.append(env.env_tick)

        if not episode % save_stats_every:
            stat_epsilon_values.append(epsilon)
            stat_mean_predators_ep_reward.append(np.mean(stat_predators_ep_reward[-save_stats_every:]))
            stat_mean_prey_ep_reward.append(np.mean(stat_prey_ep_reward[-save_stats_every:]))
            stat_mean_total_steps.append(np.mean(stat_total_steps[-save_stats_every:]))

        # save model if it improved
        if not episode % AGGREGATE_STATS_EVERY:
            # for predators
            # avg_r = round(sum(nn_save_predator_ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
            #     nn_save_predator_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            # min_r = round(min(nn_save_predator_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            # max_r = round(max(nn_save_predator_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            #
            # if avg_r >= nn_save_predator_new_average_reward:
            #     timestamp = int(time.time())
            #     for predator in predators:
            #         predator.nn.model.save(
            #             f"models/predator/{MODEL_NAME}_id{predator.id}_{avg_r}avg_{max_r}max_{min_r}min_{timestamp}.h5")
            #
            #     nn_save_predator_new_average_reward = avg_r

            # for preys
            avg_r = round(sum(nn_save_prey_ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                nn_save_prey_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            min_r = round(min(nn_save_prey_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            max_r = round(max(nn_save_prey_ep_rewards[-AGGREGATE_STATS_EVERY:]), 3)
            mean_steps = np.mean(stat_total_steps[-AGGREGATE_STATS_EVERY:])

            if mean_steps >= nn_save_prey_new_average_steps:
                timestamp = int(time.time())
                for prey in preys:
                    prey.nn.model.save(
                        f"models/prey/{MODEL_NAME}_id{prey.id}_{mean_steps}avg_steps_{avg_r}avg_{timestamp}.h5")

                nn_save_prey_new_average_steps = mean_steps

        print()
        print(f"Episode: {episode}    Steps: {env.env_tick}    Epsilon: {round(epsilon, 3)}")
        print(f"Predators reward: {round(predator_episode_reward, 3)}    Preys reward: {round(prey_episode_reward, 3)}")
        print(f"Prey training time: {timings[0]}    Predator training time: {timings[1]}")

    x = np.linspace(0, EPISODES, EPISODES // save_stats_every)

    with open(f"stats/{MODEL_NAME}_plot_epsilon", "wb") as f:
        pickle.dump(stat_epsilon_values, f)
    with open(f"stats/{MODEL_NAME}_plot_mean_total_steps", "wb") as f:
        pickle.dump(stat_mean_total_steps, f)
    with open(f"stats/{MODEL_NAME}_plot_mean_predator_rewards", "wb") as f:
        pickle.dump(stat_mean_predators_ep_reward, f)
    with open(f"stats/{MODEL_NAME}_plot_mean_prey_rewards", "wb") as f:
        pickle.dump(stat_mean_prey_ep_reward, f)
    with open(f"stats/{MODEL_NAME}_plot_x", "wb") as f:
        pickle.dump(x, f)

    fig, ax = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = ax.flatten()

    ax1.plot(x, stat_epsilon_values)
    ax1.set_title("Epsilon")

    ax2.plot(x, stat_mean_total_steps)
    ax2.set_title("Steps")

    ax3.plot(x, stat_mean_predators_ep_reward)
    ax3.set_title("Predators reward (MLP DQN)")

    ax4.plot(x, stat_mean_prey_ep_reward)
    ax4.set_title("Preys reward (CNN DQN)")

    fig.set_size_inches(12, 8)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
