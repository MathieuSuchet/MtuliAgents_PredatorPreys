import random
from math import ceil

import numpy as np
import pygame.surface

from predator_preys.entity_buffer import EntityBuffer
from predator_preys.utils import Food, Grid, Runner, MailBox, MessageHandler, MessageSender, GameConfig, \
    Leader, Follower, Howler, Decoys


class Map:
    def __init__(self, x_lims, y_lims, game_config: GameConfig):
        self.game_config = game_config
        self.lims = (x_lims, y_lims)

        self.entities = EntityBuffer()

        self.food_cap = 100
        self.food_spawn_rate = 1
        self.__cnt_food_spawn = 0

        n_food = ceil(game_config.food_config.nb_food)

        n_leaders = ceil(game_config.predator_config.nb_predators * game_config.predator_config.leader_percentage)
        n_followers = ceil(game_config.predator_config.nb_predators * game_config.predator_config.follower_percentage)

        n_howlers = ceil(game_config.prey_config.nb_preys * game_config.prey_config.howler_percentage)
        n_decoys = ceil(game_config.prey_config.nb_preys * game_config.prey_config.decoys_percentage)
        n_runners = ceil(game_config.prey_config.nb_preys * game_config.prey_config.runner_percentage)

        self.grid = Grid(self.lims)

        self.mailbox = [MailBox(), MailBox()]
        self.message_senders = [MessageSender(self.mailbox[0]), MessageSender(self.mailbox[1])]
        self.message_handlers = [MessageHandler(self.mailbox[0]), MessageHandler(self.mailbox[1])]

        self.__spawn_entities(n_food, n_leaders, n_followers, n_howlers, n_decoys, n_runners)

    def __spawn_entities(self, n_food, n_leaders, n_followers, n_howlers, n_decoys, n_runners):
        for _ in range(n_food):
            food = Food.spawn_random(*self.lims)
            self.entities.foods.append(food)
            self.grid.add_entity(food)

        for _ in range(n_leaders):
            predator = Leader.spawn_random(*self.lims)
            self.entities.predators.append(predator)
            self.grid.add_entity(predator)
            self.mailbox[0].add_entity(predator)

        for _ in range(n_followers):
            predator = Follower.spawn_random(*self.lims)
            self.entities.predators.append(predator)
            self.grid.add_entity(predator)
            self.mailbox[0].add_entity(predator)

        for _ in range(n_howlers):
            prey = Howler.spawn_random(*self.lims)
            self.entities.preys.append(prey)
            self.grid.add_entity(prey)
            self.mailbox[1].add_entity(prey)

        for _ in range(n_decoys):
            prey = Decoys.spawn_random(*self.lims)
            self.entities.preys.append(prey)
            self.grid.add_entity(prey)
            self.mailbox[1].add_entity(prey)

        for _ in range(n_runners):
            prey = Runner.spawn_random(*self.lims)
            self.entities.preys.append(prey)
            self.grid.add_entity(prey)
            self.mailbox[1].add_entity(prey)

    def update(self):
        if self.__cnt_food_spawn >= self.food_spawn_rate and len(self.entities.foods) < self.food_cap:
            food = Food.spawn_random(*self.lims)
            self.entities.foods.append(food)
            self.grid.add_entity(food)
            self.__cnt_food_spawn = 0
        self.__cnt_food_spawn += 1

        for elt in self.entities.predators:
            if elt.stamina < 0:
                elt.dead = True
            if elt.dead:
                self.entities.predators.remove(elt)
                self.grid.remove_entity(elt)
                self.mailbox[0].remove_entity(elt.id)
                continue

            if elt.can_reproduce:
                predator = random.choices(
                    [
                        Follower(*elt.position + np.random.random(size=(2,)) * 40),
                        Leader(*elt.position + np.random.random(size=(2,)) * 40)
                    ],
                    weights=[
                        self.game_config.predator_config.follower_percentage,
                        self.game_config.predator_config.leader_percentage
                    ])[0]

                predator.position[0] = np.clip(predator.position[0], 0, self.lims[0])
                predator.position[1] = np.clip(predator.position[1], 0, self.lims[1])
                self.entities.predators.append(predator)
                self.grid.add_entity(predator)
                self.mailbox[0].add_entity(predator)
                elt.can_reproduce = False

            channels = self.message_handlers[0].get_messages_for(elt.id)

            elt.update(self.grid, channels, self.message_senders[0])
            elt.position[0] = np.clip(elt.position[0], 0, self.lims[0])
            elt.position[1] = np.clip(elt.position[1], 0, self.lims[1])

            self.grid.remove_entity(elt)
            self.grid.add_entity(elt)

        for elt in self.entities.preys:
            if elt.stamina < 0:
                elt.dead = True
            if elt.dead:
                self.entities.preys.remove(elt)
                self.grid.remove_entity(elt)
                self.mailbox[1].remove_entity(elt.id)
                continue

            if elt.can_reproduce:
                prey = random.choices(
                    [
                        Runner(*elt.position + np.random.random(size=(2,)) * 40),
                        Howler(*elt.position + np.random.random(size=(2,)) * 40),
                        Decoys(*elt.position + np.random.random(size=(2,)) * 40)
                    ],
                    weights=[
                        self.game_config.prey_config.runner_percentage,
                        self.game_config.prey_config.howler_percentage,
                        self.game_config.prey_config.decoys_percentage
                    ])[0]

                prey.position[0] = np.clip(prey.position[0], 0, self.lims[0])
                prey.position[1] = np.clip(prey.position[1], 0, self.lims[1])
                self.entities.preys.append(prey)
                self.mailbox[1].add_entity(prey)
                self.grid.add_entity(prey)

                elt.can_reproduce = False

            channels = self.message_handlers[1].get_messages_for(elt.id)

            elt.update(self.grid, channels, self.message_senders[1])
            elt.position[0] = np.clip(elt.position[0], 0, self.lims[0])
            elt.position[1] = np.clip(elt.position[1], 0, self.lims[1])
            self.grid.remove_entity(elt)
            self.grid.add_entity(elt)

        for elt in self.entities.foods:
            if elt.dead:
                self.entities.foods.remove(elt)
                self.grid.remove_entity(elt)

        for m in self.mailbox:
            m.update_lifecycle()


class Renderer(object):
    DECOY_COLOR = (0, 255, 255)
    HOWLER_COLOR = (0, 128, 128)
    LEADER_COLOR = (255, 255, 0)
    FOLLOWER_COLOR = (255, 0, 0)
    RUNNER_COLOR = (0, 255, 0)

    def __init__(self):
        self.display_size = (600, 600)
        self.screen_size = (700, 700)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.init()

        pygame.display.set_caption("Prey / Predators")
        self.screen.fill((255, 255, 255))
        self.running = True
        self.font = pygame.font.SysFont('Calibri', 15)

    def render_victory(self, predator_win: bool = False):
        if predator_win:
            te = self.font.render("Predators won !", True, (0, 0, 0))
            self.screen.blit(te, (50, self.display_size[1] - 40))
        else:
            te = self.font.render("Preys won !", True, (0, 0, 0))
            self.screen.blit(te, (50, self.display_size[1] - 40))
        pygame.display.update()

    def render_draw(self):
        te = self.font.render("Draw", True, (0, 0, 0))
        self.screen.blit(te, (50, self.display_size[1] - 40))
        pygame.display.update()

    def update_display(self, map, n_steps):
        self.pygame_display(map, n_steps)
        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                self.running = False

    def draw_legend(self, legend, color, location):
        pygame.draw.circle(self.screen, color=color, center=location, radius=5)
        legend_predator = self.font.render(legend, True, (0, 0, 0))

        self.screen.blit(legend_predator, (location[0] + 10, location[1] - 8))

    def pygame_display(self, map, n_steps):
        aspect_ratio = 1.2
        self.screen.fill((255, 255, 255))
        text_preys = self.font.render(f'Number of preys : {len(map.entities.preys)}', True, (0, 0, 0))
        text_predators = self.font.render(f'Number of predators : {len(map.entities.predators)}', True,
                                          (0, 0, 0))
        text_steps = self.font.render(f'Number of steps : {n_steps}', True, (0, 0, 0))
        self.screen.blit(text_preys, (0, 0))
        self.screen.blit(text_predators, (0, 15))
        self.screen.blit(text_steps, (0, 30))

        LEGEND_OFFSET = 150

        self.draw_legend("Leader", self.LEADER_COLOR, (self.screen_size[0] / 2, 15))
        self.draw_legend("Follower", self.FOLLOWER_COLOR, (self.screen_size[0] / 2, 30))
        self.draw_legend("Runner", self.RUNNER_COLOR, (self.screen_size[0] / 2 + LEGEND_OFFSET, 15))
        self.draw_legend("Howler", self.HOWLER_COLOR, (self.screen_size[0] / 2 + LEGEND_OFFSET, 30))
        self.draw_legend("Decoy", self.DECOY_COLOR, (self.screen_size[0] / 2 + LEGEND_OFFSET, 45))

        for e in map.entities.preys:
            color = self.RUNNER_COLOR
            if isinstance(e, Decoys):
                color = self.DECOY_COLOR

            if isinstance(e, Howler):
                color = self.HOWLER_COLOR

            energy_left = self.font.render(str(e.stamina), True, (0, 0, 0))
            self.screen.blit(energy_left, (
                (e.position[0] / (map.lims[0] * aspect_ratio)) *
                self.screen_size[0] + (self.screen_size[0] - self.display_size[0]) - 20,
                (e.position[1] / (map.lims[1] * aspect_ratio)) *
                self.screen_size[1] + (self.screen_size[1] - self.display_size[1]) - 20
            ))
            pygame.draw.circle(self.screen, color=color, center=(
                (e.position[0] / (map.lims[0] * aspect_ratio)) *
                self.screen_size[0] + (self.screen_size[0] - self.display_size[0]),
                (e.position[1] / (map.lims[1] * aspect_ratio)) *
                self.screen_size[1] + (self.screen_size[1] - self.display_size[1])
            ), radius=5)
        for e in map.entities.predators:

            energy_left = self.font.render(str(e.stamina), True, (0, 0, 0))
            self.screen.blit(energy_left, (
                (e.position[0] / (map.lims[0] * aspect_ratio)) *
                self.screen_size[0] + (self.screen_size[0] - self.display_size[0]) - 20,
                (e.position[1] / (map.lims[1] * aspect_ratio)) *
                self.screen_size[1] + (self.screen_size[1] - self.display_size[1]) - 20
            ))

            color = self.FOLLOWER_COLOR
            if isinstance(e, Leader):
                color = self.LEADER_COLOR

            pygame.draw.circle(self.screen, color=color, center=(
                (e.position[0] / (map.lims[0] * aspect_ratio)) *
                self.screen_size[0] + (self.screen_size[0] - self.display_size[0]),
                (e.position[1] / (map.lims[1] * aspect_ratio)) *
                self.screen_size[1] + (self.screen_size[1] - self.display_size[1])
            ), radius=5)
        for e in map.entities.foods:
            pygame.draw.circle(self.screen, color=(0, 0, 255), center=(
                (e.position[0] / (map.lims[0] * aspect_ratio)) *
                self.screen_size[0] + (self.screen_size[0] - self.display_size[0]),
                (e.position[1] / (map.lims[1] * aspect_ratio)) *
                self.screen_size[1] + (self.screen_size[1] - self.display_size[1])
            ), radius=5)


class Manager:
    def __init__(self, game_config: GameConfig):
        self.map = Map(400, 400, game_config)
        self.time = 0
        self.steps = 0

        self.renderer = Renderer()

    def step(self):
        self.steps += 1
        self.map.update()

    def reset(self):
        self.map = Map(*self.map.lims, self.map.game_config)
        self.time = 0
        self.steps = 0
        self.renderer = Renderer()

    def close(self):
        self.map = None
        self.time = 0
        self.steps = 0
        self.renderer = Renderer()

    def render(self):
        self.renderer.update_display(self.map, self.steps)
