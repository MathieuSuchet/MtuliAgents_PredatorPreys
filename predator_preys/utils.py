import random
import time
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np


class NotEnoughSpaceOnMap(Exception):
    pass


class PreyConfig(object):
    def __init__(self, nb_preys, howler_percentage, runner_percentage, decoys_percentage):
        self.nb_preys = nb_preys
        self.howler_percentage = howler_percentage
        self.runner_percentage = runner_percentage
        self.decoys_percentage = decoys_percentage


class PredatorConfig(object):
    def __init__(self, nb_predators, leader_percentage, follower_percentage):
        self.nb_predators = nb_predators
        self.leader_percentage = leader_percentage
        self.follower_percentage = follower_percentage


class FoodConfig(object):
    def __init__(self, nb_food):
        self.nb_food = nb_food


class GameConfig(object):
    def __init__(self, prey_config: PreyConfig, predator_config: PredatorConfig, food_config: FoodConfig):
        self.prey_config = prey_config
        self.predator_config = predator_config
        self.food_config = food_config


class MessageType(Enum):
    FOLLOW_LEADER: int = 0
    PREDATOR_AT: int = 1


class Message(object):
    def __init__(self, _from: uuid.UUID, m_type: MessageType, content: Any, time_to_live: int = 1):
        self.time_to_live = time_to_live
        self.message_id = uuid.uuid4()
        self._from = _from
        self.m_type = m_type
        self.content = content


class MessageBroker(object):
    def __init__(self, user_id: uuid.UUID):
        self.user_id = user_id
        self._messages = []

    def stack_message(self, message: Message):
        self._messages.append(message)

    def pop_message(self) -> Optional[Message]:
        return self._messages[0] if self.has_message() else None

    def has_message(self) -> bool:
        return len(self._messages) > 0

    def update_lifecycle(self):
        i = 0
        while i < len(self._messages):
            message = self._messages[i]
            message.time_to_live -= 1
            if message.time_to_live == -1:
                self._messages.pop(i)
                continue

            i += 1


class MailBox(object):
    def __init__(self):
        self.broadcast_uuid = uuid.uuid4()
        self.box: Dict[uuid.UUID, MessageBroker] = {self.broadcast_uuid: MessageBroker(self.broadcast_uuid)}

    def __getitem__(self, item):
        return self.box[item]

    def init(self, entities: List[uuid.UUID]):
        for e in entities:
            self.box.setdefault(e, MessageBroker(e))

    def add_entity(self, entity):
        if entity.id in self.box.keys():
            return
        self.box.setdefault(entity.id, MessageBroker(entity.id))

    def remove_entity(self, entity_id: uuid.UUID):
        if entity_id in self.box.keys():
            self.box.pop(entity_id)

    def add_message_to(self, entity_id: uuid.UUID, message: Message):
        if entity_id in self.box.keys():
            self.box[entity_id].stack_message(message)

    def get_all_messages_of(self, _id: uuid.UUID):
        if _id not in self.box.keys():
            return None
        return self.box[_id]

    def broadcast(self, _from: uuid.UUID, message: Message):
        self.box[self.broadcast_uuid].stack_message(message)

    def get_broadcast_channel(self) -> MessageBroker:
        return self.box[self.broadcast_uuid]

    def update_lifecycle(self):
        for broker in self.box.values():
            broker.update_lifecycle()


class MessageSender(object):
    def __init__(self, mailbox: MailBox):
        self.mailbox = mailbox

    def send(self, _from: uuid.UUID, _to: uuid.UUID, _type: MessageType, _content: Any):
        self.mailbox.add_message_to(_to, Message(_from, _type, _content))

    def broadcast(self, _from: uuid.UUID, _type: MessageType, _content: Any):
        self.mailbox.broadcast(_from, Message(_from, _type, _content))


class MessageHandler(object):
    def __init__(self, mailbox: MailBox):
        self.mailbox = mailbox

    def get_messages_for(self, _dest: uuid.UUID) -> Dict[str, MessageBroker]:
        broadcast_channel = self.mailbox.get_broadcast_channel()

        brokers = {}

        if broadcast_channel.has_message():
            brokers.setdefault("broadcast", broadcast_channel)

        brokers.setdefault("self", self.mailbox[_dest])
        return brokers


class Tile(object):
    def __init__(self, tile_i):
        self._entities = []
        self.tile_i = tile_i

    def add_entity(self, entity):
        self._entities.append(entity)

    def get_entities(self):
        return self._entities

    def remove_entity(self, entity):
        self._entities.remove(entity)


class Grid(object):
    def __init__(self, total_shape):
        self.total_shape = total_shape
        self.tile_size = 20, 20

        self.nb_tiles = self.total_shape[0] // self.tile_size[0], self.total_shape[1] // self.tile_size[1]
        self.tiles = np.empty(shape=self.nb_tiles, dtype=Tile)

        for i in range(self.tiles.shape[0]):
            for j in range(self.tiles.shape[1]):
                self.tiles[i, j] = Tile((i, j))

    def add_entity(self, entity):
        entity_tile = abs(entity.position[0] - .1) // self.tile_size[0], abs(entity.position[1] - .1) // self.tile_size[
            1]
        self.tiles[int(entity_tile[0]), int(entity_tile[1])].add_entity(entity)
        entity.set_tile(self.tiles[int(entity_tile[0]), int(entity_tile[1])].tile_i)

    def remove_entity(self, entity):
        self.tiles[int(entity.tile[0]), int(entity.tile[1])].remove_entity(entity)

    def get_tile_of(self, entity):
        return (
            int(abs(entity.position[0] - .1) // self.tile_size[0]),
            int(abs(entity.position[1] - .1) // self.tile_size[1]))


class Positioned(object):
    number: int = 0
    velocity: float = 0
    dead = False

    def __init__(self, x, y):
        self.id = uuid.uuid4()
        self.stamina = 0
        self.position = np.array((x, y))
        self.tile = (-1, -1)

    def set_tile(self, tile_coords):
        self.tile = tile_coords

    @classmethod
    def spawn_random(cls, x_lim, y_lim):
        return cls(
            random.uniform(0, x_lim),
            random.uniform(0, y_lim)
        )

    def go_towards(self, elt: 'Positioned'):
        if self.dead:
            return

        same_place = False

        diff = elt.position - self.position
        if diff[0] == 0 and diff[1] == 0:
            same_place = True
        diff = (diff / np.sqrt(np.sum(diff ** 2))) if not same_place else np.zeros(shape=(2,))

        self.position += diff * self.velocity
        self.stamina -= 1

        if self.stamina <= 0:
            self.dead = True

    def run_away_from(self, elt: 'Positioned'):
        if self.dead:
            return

        same_place = False

        diff = elt.position - self.position
        if diff[0] == 0 and diff[1] == 0:
            same_place = True
        diff = (diff / np.sqrt(np.sum(diff ** 2))) if not same_place else np.zeros(shape=(2,))
        self.position -= diff * self.velocity
        self.stamina -= 1

        if self.stamina <= 0:
            self.dead = True

    def __eq__(self, other):
        if isinstance(other, Positioned):
            return self.id == other.id
        return False


class Food(Positioned):
    number: int = 1


class Predator(Positioned):
    number: int = 2
    eating_distance = 10

    def __init__(self, x, y, velocity: float = 3.0, stamina: float = 150, nb_tiles_around: int = 4):
        super().__init__(x, y)
        self.nb_tiles_around = nb_tiles_around
        self.velocity = velocity
        self.stamina = stamina
        self.dead = False
        self.can_reproduce = False
        self.stalling = False
        self.time_to_stall = 1.0
        self.cnt_time = 0.0
        self.tile = (-1, -1)

    def update(self, map: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):

        if self.stalling:
            if time.time() < self.cnt_time:
                return
            self.stalling = False

        closest_prey = None
        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= map.tiles.shape[0] or self.tile[1] + j < 0 or self.tile[
                    1] + j >= map.tiles.shape[1]:
                    continue
                for entity in map.tiles[self.tile[0] + i, self.tile[1] + j].get_entities():
                    if isinstance(entity, Prey) or issubclass(type(entity), Prey) and not entity.dead:
                        if not closest_prey or (
                                closest_prey and np.linalg.norm(entity.position - self.position) < np.linalg.norm(
                            closest_prey.position - self.position)):
                            closest_prey = entity

        if closest_prey:
            self.go_towards(closest_prey)
            if np.linalg.norm(
                    closest_prey.position - self.position) < Predator.eating_distance and not closest_prey.dead:
                closest_prey.dead = True
                self.stamina += 50
                self.can_reproduce = True
                self.cnt_time = time.time() + self.time_to_stall
                self.stalling = True

        else:
            self.stamina -= 1


class Prey(Positioned):
    number: int = 3
    eating_distance = 10

    def __init__(self, x, y, velocity: float = 3.5, stamina: float = 150, nb_tiles_around: int = 3):
        super().__init__(x, y)
        self.nb_tiles_around = nb_tiles_around
        self.velocity = velocity
        self.stamina = stamina
        self.dead = False
        self.can_reproduce = False
        self.tile = (-1, -1)

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        pass

    def try_to_eat(self, food: Food):
        if np.linalg.norm(food.position - self.position) < self.eating_distance:
            food.dead = True
            self.stamina += 50
            self.can_reproduce = True


class Runner(Prey):
    """
    Prey that runs away from predators
    """

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        closest_predator = None
        closest_food = None

        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= grid.tiles.shape[0] or self.tile[1] + j < 0 or self.tile[1] + j >= grid.tiles.shape[1]:
                    continue
                for entity in grid.tiles[self.tile[0] + i, self.tile[1] + j].get_entities():
                    if isinstance(entity, Predator) or issubclass(type(entity), Predator):
                        if (
                                (closest_predator
                                 and np.linalg.norm(closest_predator.position - self.position) > np.linalg.norm(
                                            entity.position - self.position)) or not closest_predator) \
                                or not closest_predator:
                            closest_predator = entity

                    if isinstance(entity, Food):
                        if not closest_food or \
                                (closest_food and np.linalg.norm(
                                    closest_food.position - self.position) > np.linalg.norm(
                                    entity.position - self.position)):
                            closest_food = entity

        # we got the closest predator and food, choice

        # we got none around
        if not closest_food and not closest_predator:
            # wander around
            direction = np.random.uniform(-1, 1, size=(2,))
            direction /= np.linalg.norm(direction)

            self.position += direction * self.velocity

        elif closest_predator and not closest_food or (closest_food and closest_predator and np.linalg.norm(
                closest_food.position - self.position) >= np.linalg.norm(closest_predator.position - self.position)):
            # Run
            self.run_away_from(closest_predator)

        elif ((closest_food and not closest_predator)
              or (
                      closest_food
                      and closest_predator
                      and np.linalg.norm(closest_predator.position - self.position) > np.linalg.norm(
                  closest_food.position - self.position)
              )):
            # Go towards
            self.go_towards(closest_food)
            self.try_to_eat(closest_food)


class Howler(Prey):
    """
    Prey that howls to alert other preys there are predators nearby
    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        entities_to_howl_at = []
        predators_to_howl = []
        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= grid.tiles.shape[0] or self.tile[1] + j < 0 or \
                        self.tile[1] + j >= grid.tiles.shape[1]:
                    continue

                t = grid.tiles[self.tile[0] + i, self.tile[1] + j]
                for entity in t.get_entities():
                    if isinstance(entity, Prey):
                        entities_to_howl_at.append(entity.id)
                    if isinstance(entity, Predator):
                        predators_to_howl.append(entity.position)

        for p in predators_to_howl:
            for e in entities_to_howl_at:
                sender.send(self.id, e, MessageType.PREDATOR_AT, p)






class Decoys(Prey):
    """
    Prey that can outrun predators and keep them close
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        self.velocity += 2
        self.stamina *= 1.5

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        if "self" in channels.keys() and channels["self"].has_message():
            message = channels["self"].pop_message()

            if message.m_type == MessageType.PREDATOR_AT:
                predator_position = message.content[1]

                direction = self.position - predator_position

                if np.linalg.norm(direction) < 50:
                    # Good
                    super().update(grid, channels, sender)
                    return

                direction += direction * 0.1
                direction /= np.linalg.norm(direction)

                self.position += direction * self.velocity
                self.stamina -= 1
        else:
            closest_predator = None
            closest_food = None

            for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                    if self.tile[0] + i < 0 or self.tile[0] + i >= grid.tiles.shape[0] or self.tile[1] + j < 0 or \
                            self.tile[
                                1] + j >= grid.tiles.shape[1]:
                        continue
                    for entity in grid.tiles[self.tile[0] + i, self.tile[1] + j].get_entities():
                        if isinstance(entity, Predator) or issubclass(type(entity), Predator):
                            if (
                                    (closest_predator
                                     and np.linalg.norm(closest_predator.position - self.position) > np.linalg.norm(
                                                entity.position - self.position)) or not closest_predator) \
                                    or not closest_predator:
                                closest_predator = entity

                        if isinstance(entity, Food):
                            if not closest_food or \
                                    (closest_food and np.linalg.norm(
                                        closest_food.position - self.position) > np.linalg.norm(
                                        entity.position - self.position)):
                                closest_food = entity

            if closest_predator and closest_food:
                if np.linalg.norm(self.position - closest_predator.position) < np.linalg.norm(
                        self.position - closest_food.position):
                    closest_food = None
                else:
                    closest_predator = None

            if closest_predator and not closest_food:
                if np.linalg.norm(self.position - closest_predator.position) > 30:
                    self.go_towards(closest_predator)
                else:
                    self.run_away_from(closest_predator)

            if not closest_predator and closest_food:
                self.go_towards(closest_food)
                self.try_to_eat(closest_food)


class Leader(Predator):
    """
    Predator that leads a group of predator
    """

    def update(self, map: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        entities_to_update = []
        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= map.tiles.shape[0] or self.tile[1] + j < 0 or self.tile[
                    1] + j >= map.tiles.shape[1]:
                    continue

                t = map.tiles[i][j]
                for e in t.get_entities():
                    if isinstance(e, Predator):
                        entities_to_update.append(e.id)

        for entity in entities_to_update:
            sender.send(self.id, entity, MessageType.FOLLOW_LEADER, [self.id, self.position])

        super().update(map, channels, sender)


class Follower(Predator):
    """
    Predator that follows a leader
    """

    def update(self, map: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        if "self" in channels.keys() and channels["self"].has_message():
            message = channels["self"].pop_message()
            if message.m_type == MessageType.FOLLOW_LEADER:
                leader_pos = message.content[1]

                direction = leader_pos - self.position

                if np.linalg.norm(direction) < 10:
                    # Good
                    return

                direction -= direction * 0.1
                direction /= np.linalg.norm(direction)

                self.position += direction * self.velocity
                self.stamina -= 1
        else:
            super().update(map, channels, sender)
