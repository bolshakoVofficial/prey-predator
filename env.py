import numpy as np
import random
import pygame as pg
import cv2
from agents import Predator, Prey
from collections import deque


class Env:
    def __init__(self, screen_size=(1000, 1000), env_size=(100, 100), scale=10, draw_env_window=True,
                 raw_map_view=False, predators_number=2, preys_number=1, max_env_ticks=300):
        self.screen_size = screen_size
        self.env_size = env_size
        self.max_dist = (self.env_size[0] ** 2 + self.env_size[1] ** 2) ** 0.5
        self.scale = scale
        self.x_scaled = self.screen_size[0] // self.env_size[0]
        self.y_scaled = self.screen_size[1] // self.env_size[1]
        self.draw_env_window = draw_env_window
        self.raw_map_view = raw_map_view

        self.predator_size = (50, 50)
        self.prey_size = (75, 75)
        self.predators_number = predators_number
        self.preys_number = preys_number

        self.frame = 5
        self.pixel_history_len = 10
        self.pixel_state_shape = (self.env_size[0] - 2 * self.frame,
                                  self.env_size[1] - 2 * self.frame,
                                  self.pixel_history_len)
        self.predator_state_shape = (2 * self.predators_number + 2 * self.preys_number,)
        self.prey_state_shape = (2 * self.predators_number + 2 * self.preys_number,)

        self.predator_speed = 2
        self.prey_speed = 3
        self.attack_distance = self.predator_speed * 2
        self.prey_hunt_reward = 10
        self.prey_death_penalty = -10

        self.preys = None
        self.predators = []
        self.predators_last_action = None
        self.last_state_positions = None

        self.cells = {
            "obstacle": 0,
            "empty": 1,
            "predator": 2,
            "prey": 3,
        }

        self.env_tick = 0
        self.max_env_ticks = max_env_ticks
        self.obstacles = self.place_obstacles()

        self.preys = [Prey(i, None, self) for i in range(self.preys_number)]
        self.predators = [Predator(i, None, self) for i in range(self.predators_number)]

        self.predator_n_actions = self.predators[0].n_actions
        self.prey_n_actions = self.preys[0].n_actions

        self.preys_cells, self.predators_cells = self.reset_actors()

        self.env_map = self.build_env()

        # Colors
        self.colors = {"black": (0, 0, 0), "red": (255, 0, 0),
                       "green": (0, 255, 0), "fancy": (150, 0, 255),
                       "purple": (70, 0, 70), "yellow": (255, 255, 0)}

        self.pixel_history = deque(maxlen=self.pixel_history_len)
        self.pixel_history.extend([self.get_pixel_state() for _ in range(self.pixel_history_len)])

        # Images initialization
        self.env_image = pg.image.load("icons/env.png")
        self.env_image = pg.transform.scale(self.env_image, self.screen_size)

        self.prey_image = pg.image.load("icons/antelope.png")
        self.prey_image = pg.transform.scale(self.prey_image, self.prey_size)

        self.predator_image = pg.image.load("icons/lion.png")
        self.predator_image = pg.transform.scale(self.predator_image, self.predator_size)

    def increment_env_tick(self):
        self.env_tick += 1

    def place_obstacles(self):
        obstacles = []
        for i in (list(range(self.frame)) + list(range(self.env_size[0] - self.frame, self.env_size[0]))):
            for j in range(self.env_size[1]):
                obstacles.append((i, j))

        for i in range(self.env_size[0]):
            for j in (list(range(self.frame)) + list(range(self.env_size[1] - self.frame, self.env_size[1]))):
                obstacles.append((i, j))

        return obstacles

    def reset_actors(self):
        prey_shift = 20 + self.frame
        predator_shift = 10 + self.frame

        predator_spawn_1 = (predator_shift, predator_shift)
        predator_spawn_2 = (predator_shift, self.env_size[0] - predator_shift)
        predator_spawn_3 = (self.env_size[0] - predator_shift, self.env_size[1] - predator_shift)
        predator_spawn_4 = (self.env_size[0] - predator_shift, predator_shift)

        predator_spawns = [predator_spawn_1, predator_spawn_2, predator_spawn_3, predator_spawn_4]

        prey_spawn_1 = (self.env_size[0] // 2, prey_shift)
        prey_spawn_2 = (self.env_size[0] // 2, self.env_size[1] - prey_shift)
        prey_spawn_3 = (prey_shift, self.env_size[1] // 2)
        prey_spawn_4 = (self.env_size[0] - prey_shift, self.env_size[1] // 2)
        prey_spawn_5 = (self.env_size[0] // 2, self.env_size[1] // 2)

        prey_spawns = [prey_spawn_1, prey_spawn_2, prey_spawn_3, prey_spawn_4, prey_spawn_5]

        preys_cells = []
        predators_cells = []

        for prey in self.preys:
            prey.set_alive_status()
            prey_pos = (np.random.randint(-5, 5), np.random.randint(-5, 5))
            prey.position = np.add(random.choice(prey_spawns), prey_pos)
            preys_cells.append(prey.position)

        for predator in self.predators:
            predator_pos = (np.random.randint(-5, 5), np.random.randint(-5, 5))
            predator.position = np.add(random.choice(predator_spawns), predator_pos)
            predators_cells.append(predator.position)

        return preys_cells, predators_cells

    def build_env(self):
        environment = np.full(self.env_size, self.cells['empty'])

        for cell in self.obstacles:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['obstacle']

        for cell in self.preys_cells:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['prey']

        for cell in self.predators_cells:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['predator']

        return environment

    def step(self, predator_actions, prey_actions):
        self.last_state_positions = []

        for i, predator in enumerate(self.predators):
            self.last_state_positions.append(predator.position)

            # predator.move(predator_actions[i])
            predator.cheat_move(self)

        for i, prey in enumerate(self.preys):
            if prey.is_alive():
                self.last_state_positions.append(prey.position)

                # prey.run_away(self)
                prey.move(prey_actions[i])

        predator_reward = -1  # penalty per step
        prey_rewards = {}
        terminated = False

        for prey in self.preys:
            if prey.is_alive():
                distances = [self.distance_between_units(prey.position, predator.position) for predator in
                             self.predators]
                if min(distances) <= self.attack_distance:
                    prey.set_not_alive_status()
                    predator_reward += self.prey_hunt_reward
                    prey_rewards[prey.id] = self.prey_death_penalty
                else:
                    prey_rewards[prey.id] = min(distances) / self.max_dist
            else:
                prey.set_not_alive_status()
                prey_rewards[prey.id] = 0

        preys_alive = [prey.id for prey in self.preys if prey.is_alive()]

        if not preys_alive or (self.env_tick >= self.max_env_ticks):
            terminated = True

        self.update_map()
        self.predators_last_action = predator_actions
        self.pixel_history.append(self.get_pixel_state())

        return predator_reward, prey_rewards, terminated

    def reset(self):
        self.env_tick = 0
        self.obstacles = self.place_obstacles()
        self.preys_cells, self.predators_cells = self.reset_actors()
        self.update_map()
        self.pixel_history.extend([self.get_pixel_state() for _ in range(self.pixel_history_len)])

        # last actions = no_op actions when reset
        self.predators_last_action = [8 for _ in range(len(self.predators))]

    def get_state(self):
        # state = [len(self.preys), len(self.predators)]
        state = []

        center_x = self.env_size[0] / 2
        center_y = self.env_size[1] / 2

        for predator in self.predators:
            state.append((predator.position[0] - center_x) / self.env_size[0])
            state.append((predator.position[1] - center_y) / self.env_size[1])

        for prey in self.preys:
            state.append((prey.position[0] - center_x) / self.env_size[0])
            state.append((prey.position[1] - center_y) / self.env_size[1])

        prey_state = state.copy()
        # state.extend(self.predators_last_action)

        return np.array(state), np.array(prey_state)

    def get_pixel_state(self):
        feature_map = np.zeros((self.env_size[0], self.env_size[1], 3), np.uint8)

        for i, predator in enumerate(self.predators):
            cv2.line(feature_map,
                     tuple(np.add(predator.position, (-3, 0))),
                     tuple(np.add(predator.position, (3, 0))),
                     self.colors["red"], 1)
            cv2.line(feature_map,
                     tuple(np.add(predator.position, (0, -3))),
                     tuple(np.add(predator.position, (0, 3))),
                     self.colors["red"], 1)

        alive_preys = [prey for prey in self.preys if prey.is_alive()]
        for i, prey in enumerate(alive_preys, start=len(self.predators)):
            cv2.line(feature_map,
                     tuple(np.add(prey.position, (0, 3))),
                     tuple(np.add(prey.position, (3, -2))),
                     self.colors["green"], 1)
            cv2.line(feature_map,
                     tuple(np.add(prey.position, (0, 3))),
                     tuple(np.add(prey.position, (-3, -2))),
                     self.colors["green"], 1)
            cv2.line(feature_map,
                     tuple(np.add(prey.position, (-3, -2))),
                     tuple(np.add(prey.position, (3, -2))),
                     self.colors["green"], 1)
            cv2.circle(feature_map, tuple(prey.position), 2, self.colors["green"], -1)

        feature_map = feature_map[self.frame:self.env_size[0] - self.frame, self.frame:self.env_size[1] - self.frame]
        feature_map = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)

        return np.array(feature_map)

    def get_pixel_history(self):
        return np.dstack(self.pixel_history)

    def update_map(self):
        preys_cells = []
        predators_cells = []

        for prey in self.preys:
            if prey.is_alive():
                preys_cells.append(prey.position)

        for predator in self.predators:
            predators_cells.append(predator.position)

        environment = np.full(self.env_size, self.cells['empty'])

        for cell in self.obstacles:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['obstacle']

        for cell in preys_cells:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['prey']

        for cell in predators_cells:
            environment[cell[0] - 1][cell[1] - 1] = self.cells['predator']

        self.env_map = environment

    def draw_env(self, screen: pg.display.set_mode((10, 10))):
        screen.blit(self.env_image, (0, 0))

        for prey in self.preys:
            if prey.is_alive():
                scaled_position = (prey.position[0] * self.x_scaled - self.prey_size[0] // 2,
                                   prey.position[1] * self.y_scaled - self.prey_size[1] // 2)
                screen.blit(self.prey_image, scaled_position)

        for predator in self.predators:
            scaled_position = (predator.position[0] * self.x_scaled - self.predator_size[0] // 2,
                               predator.position[1] * self.y_scaled - self.predator_size[1] // 2)
            screen.blit(self.predator_image, scaled_position)

        if self.raw_map_view:
            for x in range(self.env_size[0]):
                for y in range(self.env_size[1]):
                    if self.env_map[x][y] == self.cells["obstacle"]:  # obstacle
                        pg.draw.rect(screen, self.colors["black"],
                                     [x * self.x_scaled, y * self.y_scaled, self.scale, self.scale])
                    elif self.env_map[x][y] == self.cells["prey"]:  # prey
                        pg.draw.rect(screen, self.colors["green"],
                                     [x * self.x_scaled, y * self.y_scaled, self.scale, self.scale])
                    elif self.env_map[x][y] == self.cells["predator"]:  # predator
                        pg.draw.rect(screen, self.colors["red"],
                                     [x * self.x_scaled, y * self.y_scaled, self.scale, self.scale])

                    if not (x % 10) and not (y % 10):
                        pg.draw.rect(screen, self.colors["fancy"],
                                     [x * self.x_scaled, y * self.y_scaled, self.scale, self.scale])

    @staticmethod
    def distance_between_units(pos1, pos2):
        return ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5
