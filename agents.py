import numpy as np
import random
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class Agent:
    def __init__(self, id, position, env):
        self.id = id
        self.position = position
        self.cells = env.cells
        self.env = env
        self.speed = 1

    def move(self, direction):
        dx, dy = 0, 0
        if direction == 0:
            dy = -1
        elif direction == 1:
            dy = -1
            dx = 1
        elif direction == 2:
            dx = 1
        elif direction == 3:
            dy = 1
            dx = 1
        elif direction == 4:
            dy = 1
        elif direction == 5:
            dy = 1
            dx = -1
        elif direction == 6:
            dx = -1
        elif direction == 7:
            dy = -1
            dx = -1
        elif direction == 8:
            pass  # no_op action

        x, y = self.position[0], self.position[1]
        x += dx * self.speed
        y += dy * self.speed

        if x < 0:
            x = 0
        if x > len(self.env.env_map[0]):
            x = len(self.env.env_map[0]) - 1
        if y < 0:
            y = 0
        if y > len(self.env.env_map[1]):
            y = len(self.env.env_map[1]) - 1

        self.position = x, y

    def available_actions(self, env_map):
        act0, act1, act2, act3 = 0, 0, 0, 0
        act4, act5, act6, act7 = 0, 0, 0, 0
        up = True
        right = True
        down = True
        left = True

        if (self.position[0] - self.speed) < self.env.frame:
            left = False
        if (self.position[0] + self.speed) > (len(env_map[0]) - self.env.frame):
            right = False
        if (self.position[1] - self.speed) < self.env.frame:
            up = False
        if (self.position[1] + self.speed) > (len(env_map[1]) - self.env.frame):
            down = False

        blocked = [self.cells['obstacle'], self.cells['predator'], self.cells['prey']]

        if up and not (env_map[self.position[0]][self.position[1] - self.speed] in blocked):
            act0 = 1
        if up and right and not (env_map[self.position[0] + self.speed][self.position[1] - self.speed] in blocked):
            act1 = 1
        if right and not (env_map[self.position[0] + self.speed][self.position[1]] in blocked):
            act2 = 1
        if right and down and not (env_map[self.position[0] + self.speed][self.position[1] + self.speed] in blocked):
            act3 = 1
        if down and not (env_map[self.position[0]][self.position[1] + self.speed] in blocked):
            act4 = 1
        if left and down and not (env_map[self.position[0] - self.speed][self.position[1] + self.speed] in blocked):
            act5 = 1
        if left and not (env_map[self.position[0] - self.speed][self.position[1]] in blocked):
            act6 = 1
        if left and up and not (env_map[self.position[0] - self.speed][self.position[1] - self.speed] in blocked):
            act7 = 1
        return [act0, act1, act2, act3, act4, act5, act6, act7, 1]


class Predator(Agent):
    def __init__(self, id, position, env):
        super().__init__(id, position, env)
        self.speed = env.predator_speed
        self.path = []
        self.path_stage = 0
        self.n_actions = 9
        self.nn = None

    @staticmethod
    def find_path(coords, new_coords, env_map):
        matrix = np.array(env_map).T
        grid = Grid(matrix=matrix)
        start = grid.node(coords[0], coords[1])
        end = grid.node(new_coords[0], new_coords[1])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, _ = finder.find_path(start, end, grid)
        return path

    def cheat_move(self, env):
        try:
            alive_preys = [prey for prey in env.preys if prey.is_alive()]
            self.path = self.find_path(self.position, alive_preys[0].position, env.env_map)
            self.path_stage = (1 * self.speed) if len(self.path) > self.speed else 1
            self.position = self.path[self.path_stage]
        except:
            pass


class Prey(Agent):
    def __init__(self, id, position, env):
        super().__init__(id, position, env)

        self.alive_status = True
        self.steps_not_alive = 0
        self.speed = env.prey_speed
        self.n_actions = 9
        self.nn = None

    def calm_walk(self, direction):
        dx, dy = 0, 0
        if direction == 0:
            dy = -1
        elif direction == 1:
            dy = -1
            dx = 1
        elif direction == 2:
            dx = 1
        elif direction == 3:
            dy = 1
            dx = 1
        elif direction == 4:
            dy = 1
        elif direction == 5:
            dy = 1
            dx = -1
        elif direction == 6:
            dx = -1
        elif direction == 7:
            dy = -1
            dx = -1
        elif direction == 8:
            pass  # no_op action

        x, y = self.position[0], self.position[1]
        x += dx
        y += dy

        if x < 0:
            x = 0
        if x > len(self.env.env_map[0]):
            x = len(self.env.env_map[0]) - 1
        if y < 0:
            y = 0
        if y > len(self.env.env_map[1]):
            y = len(self.env.env_map[1]) - 1

        self.position = x, y

    def run_away(self, env):
        better_run = False
        for predator in env.predators:
            if env.distance_between_units(self.position, predator.position) < env.predator_speed * 10:
                better_run = True

        if better_run:
            pass
        else:
            avail_actions = self.available_actions(env.env_map)
            avail_actions_index = np.nonzero(avail_actions)[0]
            action = random.choice(avail_actions_index)
            self.calm_walk(action)

    def is_alive(self):
        return self.alive_status

    def how_long_not_alive(self):
        return self.steps_not_alive

    def set_not_alive_status(self):
        self.alive_status = False
        self.steps_not_alive += 1

    def set_alive_status(self):
        self.alive_status = True
        self.steps_not_alive = 0

    def find_closest_enemy(self, env):
        pass
