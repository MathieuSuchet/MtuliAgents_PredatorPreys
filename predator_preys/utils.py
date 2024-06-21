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
        """
        Configuration des proies
        :param nb_preys: Nombre de proies
        :param howler_percentage: Pourcentage de hurleurs
        :param runner_percentage: Pourcentage de coureurs
        :param decoys_percentage: Pourcentage de leurres
        """
        self.nb_preys = nb_preys
        self.howler_percentage = howler_percentage
        self.runner_percentage = runner_percentage
        self.decoys_percentage = decoys_percentage


class PredatorConfig(object):
    def __init__(self, nb_predators, leader_percentage, follower_percentage):
        """
        Configuration des prédateurs
        :param nb_predators: Nombre de prédateurs
        :param leader_percentage: Nombre de chefs de meutes
        :param follower_percentage: Nombre de suiveurs
        """
        self.nb_predators = nb_predators
        self.leader_percentage = leader_percentage
        self.follower_percentage = follower_percentage


class FoodConfig(object):
    def __init__(self, nb_food):
        """
        Configuration de la nourriture
        :param nb_food: Nombre de nourriture
        """
        self.nb_food = nb_food


class GameConfig(object):
    def __init__(self, prey_config: PreyConfig, predator_config: PredatorConfig, food_config: FoodConfig):
        """
        Configuration du jeu
        :param prey_config: Configuration des proies
        :param predator_config: Configuration des prédateurs
        :param food_config: Configuration de la nourriture
        """
        self.prey_config = prey_config
        self.predator_config = predator_config
        self.food_config = food_config


class MessageType(Enum):
    """
    Type de message
    """
    FOLLOW_LEADER: int = 0
    PREDATOR_AT: int = 1


class Message(object):
    def __init__(self, _from: uuid.UUID, m_type: MessageType, content: Any, time_to_live: int = 1):
        """
        Message
        :param _from: ID de l'entité envoyant le message
        :param m_type: Type du message
        :param content: Contenu du message
        :param time_to_live: Temps de vie
        """
        self.time_to_live = time_to_live
        self.message_id = uuid.uuid4()
        self._from = _from
        self.m_type = m_type
        self.content = content


class MessageBroker(object):
    def __init__(self, user_id: uuid.UUID):
        """
        File des messages de l'entité
        :param user_id: ID de l'entité
        """
        self.user_id = user_id
        self._messages = []

    def stack_message(self, message: Message):
        """
        Ajoute un message à la file
        :param message: Le message à ajouter
        """
        self._messages.append(message)

    def pop_message(self) -> Optional[Message]:
        """
        Retourne le premier message de la file
        :return: Le premier message de la file
        """
        return self._messages[0] if self.has_message() else None

    def has_message(self) -> bool:
        """
        Vérifie s'il y a des messages dans le file
        :return: True si oui, False sinon
        """
        return len(self._messages) > 0

    def update_lifecycle(self):
        """
        Mise à jour du temps de vie des messages
        """
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
        """
        Boite aux lettres contenant les chaines de toutes les entités
        """
        self.broadcast_uuid = uuid.uuid4()
        self.box: Dict[uuid.UUID, MessageBroker] = {self.broadcast_uuid: MessageBroker(self.broadcast_uuid)}

    def __getitem__(self, item):
        return self.box[item]

    def init(self, entities: List[uuid.UUID]):
        for e in entities:
            self.box.setdefault(e, MessageBroker(e))

    def add_entity(self, entity):
        """
        Ajoute une entité à la boite
        :param entity: Entité à ajouter
        """
        if entity.id in self.box.keys():
            return
        self.box.setdefault(entity.id, MessageBroker(entity.id))

    def remove_entity(self, entity_id: uuid.UUID):
        """
        Enlever une entité de la boite
        :param entity_id: Entité à enlever
        """
        if entity_id in self.box.keys():
            self.box.pop(entity_id)

    def add_message_to(self, entity_id: uuid.UUID, message: Message):
        """
        Ajouter un message à la chaine de l'entité choisie
        :param entity_id: Entité choisie
        :param message: Message à envoyer
        :return:
        """
        if entity_id in self.box.keys():
            self.box[entity_id].stack_message(message)

    def get_all_messages_of(self, _id: uuid.UUID):
        """
        Retourne tous les messages d'une entité
        :param _id: ID de l'entité
        """
        if _id not in self.box.keys():
            return None
        return self.box[_id]

    def broadcast(self, _from: uuid.UUID, message: Message):
        """
        Envoie un message dans le salon de broadcast
        :param _from: L'entité source
        :param message: Le message à envoyer
        """
        self.box[self.broadcast_uuid].stack_message(message)

    def get_broadcast_channel(self) -> MessageBroker:
        """
        Récupérer la chaine de broadcast
        :return: La chaine de broadcast
        """
        return self.box[self.broadcast_uuid]

    def update_lifecycle(self):
        """
        Mettre à jour le cycle de vie des messages
        """
        for broker in self.box.values():
            broker.update_lifecycle()


class MessageSender(object):
    def __init__(self, mailbox: MailBox):
        """
        Interface permettant d'envoyer des messages
        :param mailbox: La boite aux lettres
        """
        self.mailbox = mailbox

    def send(self, _from: uuid.UUID, _to: uuid.UUID, _type: MessageType, _content: Any):
        """
        Envoie un message à une entité
        :param _from: Entité source
        :param _to: Entité destination
        :param _type: Type du message
        :param _content: Contenu du message
        :return:
        """
        self.mailbox.add_message_to(_to, Message(_from, _type, _content))

    def broadcast(self, _from: uuid.UUID, _type: MessageType, _content: Any):
        """
        Envoie un message dans le broadcast
        :param _from: Entité source
        :param _type: Type de message
        :param _content: Contenu du message
        :return:
        """
        self.mailbox.broadcast(_from, Message(_from, _type, _content))


class MessageHandler(object):
    def __init__(self, mailbox: MailBox):
        """
        Interface permettant de recevoir des messages
        :param mailbox: Boite aux lettres
        """
        self.mailbox = mailbox

    def get_messages_for(self, _dest: uuid.UUID) -> Dict[str, MessageBroker]:
        """
        Récupère les chaines de l'entité et de broadcast si celle-ci a des messages
        :param _dest: Entité concernée
        :return: Un dictionnaire contant le broadcast et les chaines
        """
        broadcast_channel = self.mailbox.get_broadcast_channel()

        brokers = {}

        if broadcast_channel.has_message():
            brokers.setdefault("broadcast", broadcast_channel)

        brokers.setdefault("self", self.mailbox[_dest])
        return brokers


class Tile(object):
    def __init__(self, tile_i):
        """
        Case sur la carte
        :param tile_i: Index de la case
        """
        self._entities = []
        self.tile_i = tile_i

    def add_entity(self, entity):
        """
        Ajoute une entité à la case
        :param entity: Entité à ajouter
        """
        self._entities.append(entity)

    def get_entities(self):
        """
        Retourne les entités de la case
        :return: Entités de la case
        """
        return self._entities

    def remove_entity(self, entity):
        """
        Supprime une entité de la case
        :param entity: Entité à supprimer
        """
        self._entities.remove(entity)


class Grid(object):
    def __init__(self, total_shape):
        """
        Grille de la carte
        :param total_shape: Dimensions de la carte
        """
        self.total_shape = total_shape
        self.tile_size = 20, 20

        self.nb_tiles = self.total_shape[0] // self.tile_size[0], self.total_shape[1] // self.tile_size[1]
        self.tiles = np.empty(shape=self.nb_tiles, dtype=Tile)

        for i in range(self.tiles.shape[0]):
            for j in range(self.tiles.shape[1]):
                self.tiles[i, j] = Tile((i, j))

    def add_entity(self, entity):
        """
        Ajoute une entité à la grille
        :param entity: Entité à ajouter
        """
        entity_tile = abs(entity.position[0] - .1) // self.tile_size[0], abs(entity.position[1] - .1) // self.tile_size[
            1]
        self.tiles[int(entity_tile[0]), int(entity_tile[1])].add_entity(entity)
        entity.set_tile(self.tiles[int(entity_tile[0]), int(entity_tile[1])].tile_i)

    def remove_entity(self, entity):
        """
        Supprime une entité de la grille
        :param entity: Entité à supprimer
        """
        self.tiles[int(entity.tile[0]), int(entity.tile[1])].remove_entity(entity)

    def get_tile_of(self, entity):
        """
        Retourne la Tile où se situe une entité
        :param entity: Entité concernée
        :return: L'index de la Tile
        """
        return (
            int(abs(entity.position[0] - .1) // self.tile_size[0]),
            int(abs(entity.position[1] - .1) // self.tile_size[1]))


class Positioned(object):
    number: int = 0
    velocity: float = 0
    dead = False

    def __init__(self, x, y):
        """
        Entité positionnable
        :param x: Coordonnées x
        :param y: Coordonnées y
        """
        self.id = uuid.uuid4()
        self.stamina = 0
        self.position = np.array((x, y))
        self.tile = (-1, -1)

    def set_tile(self, tile_coords):
        self.tile = tile_coords

    @classmethod
    def spawn_random(cls, x_lim, y_lim):
        """
        Fais apparaitre de manière aléatoire une entité
        :param x_lim: Limites de spawn sur x
        :param y_lim: Limites de spawn sur y
        :return: L'entité crée
        """
        return cls(
            random.uniform(0, x_lim),
            random.uniform(0, y_lim)
        )

    def go_towards(self, elt: 'Positioned'):
        """
        Aller en direction d'une entité
        :param elt: L'entité vers laquelle aller
        """
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
        """
        Fuir une entité
        :param elt: L'entité à fuir
        """
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
    """
    Nourriture
    """
    number: int = 1


class Predator(Positioned):
    number: int = 2
    eating_distance = 10

    def __init__(self, x, y, velocity: float = 3.0, stamina: float = 150, nb_tiles_around: int = 4):
        """
        Prédateur de base
        :param x: Coordonnées x
        :param y: Coordonnées y
        :param velocity: Velocité du prédateur
        :param stamina:  Endurance du prédateur
        :param nb_tiles_around: Champ de vision
        """
        super().__init__(x, y)
        self.nb_tiles_around = nb_tiles_around
        self.velocity = velocity
        self.stamina = stamina
        self.dead = False
        self.can_reproduce = False
        self.stalling = False
        self.time_to_stall = 1.0
        self.cnt_time = 0.0

        self.picked_dir = np.zeros(shape=(2, ))
        self.time_to_wander = 30
        self._cnt_ttw = 30

        self.tile = (-1, -1)

    def wander_around(self):
        """
        Activité lorsqu'il n'y a rien à faire
        """

        if self._cnt_ttw < self.time_to_wander:
            self.position += self.picked_dir * self.velocity
            self._cnt_ttw += 1
        else:
            self.picked_dir = np.random.uniform(high=1, low=-1, size=(2, ))
            self.picked_dir /= np.linalg.norm(self.picked_dir)
            self._cnt_ttw = 0

        self.stamina -= 1

    def act(self, closest_prey):
        """
        Activité lorsqu'il y a une proie
        :param closest_prey: La proie visée
        """

        if not closest_prey:
            self.wander_around()
        else:
            self.go_towards(closest_prey)
            if np.linalg.norm(
                    closest_prey.position - self.position) < Predator.eating_distance and not closest_prey.dead:
                closest_prey.dead = True
                self.stamina += 50
                self.can_reproduce = True
                self.cnt_time = time.time() + self.time_to_stall
                self.stalling = True

    def update(self, map: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        """
        Mise à jour logique du prédateur
        :param map: La grille
        :param channels: Ses chaines (pour consulter ses messages)
        :param sender: Interface d'envoi
        """

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

        self.act(closest_prey)


class Prey(Positioned):
    number: int = 3
    eating_distance = 10

    def __init__(self, x, y, velocity: float = 3.5, stamina: float = 150, nb_tiles_around: int = 3):
        """
        Proie de base
        :param x: Coordonnées x
        :param y: Coordonnées y
        :param velocity: Vélocité de la proie
        :param stamina: Endurance de la proie
        :param nb_tiles_around: Champ de vision
        """
        super().__init__(x, y)
        self.nb_tiles_around = nb_tiles_around
        self.velocity = velocity
        self.stamina = stamina
        self.dead = False
        self.can_reproduce = False
        self.tile = (-1, -1)

        self.picked_dir = np.zeros(shape=(2,))
        self.time_to_wander = 30
        self._cnt_ttw = 30

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        """
        Mise à jour logique de la proie
        :param grid: La grille
        :param channels: Ses chaines (pour traiter ses messages)
        :param sender: L'interface d'envoi de messages
        """
        pass

    def wander_around(self):
        """
        Activité s'il n'y a pas de prédateur ou nourriture
        """

        if self._cnt_ttw < self.time_to_wander:
            self.position += self.picked_dir * self.velocity
            self._cnt_ttw += 1
        else:
            self.picked_dir = np.random.uniform(high=1, low=-1, size=(2,))
            self.picked_dir /= np.linalg.norm(self.picked_dir)
            self._cnt_ttw = 0

        self.stamina -= 1

    def try_to_eat(self, food: Food):
        """
        Essaye de manger la nourriture
        :param food: La nourriture à manger
        """
        if np.linalg.norm(food.position - self.position) < self.eating_distance:
            food.dead = True
            self.stamina += 50
            self.can_reproduce = True

    def act(self, closest_predator, closest_food):
        """
        Activité en cas de prédateur ou nourriture
        :param closest_predator: Prédateur le plus proche
        :param closest_food: Nourriture la plus proche
        :return:
        """
        if not closest_food and not closest_predator:
            # wander around
            self.wander_around()

        elif closest_predator and not closest_food or (closest_food and closest_predator and np.linalg.norm(
                closest_food.position - self.position) >= np.linalg.norm(
            closest_predator.position - self.position)):
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


class Runner(Prey):
    """
    Prey that runs away from predators
    """

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        closest_predator = None
        closest_food = None

        if "self" in channels and channels["self"].has_message():
            message = channels["self"].pop_message()

            if message.m_type == MessageType.PREDATOR_AT:
                closest_predator = Predator(*message.content)

        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= grid.tiles.shape[0] or self.tile[1] + j < 0 or self.tile[
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

        self.act(closest_predator, closest_food)


class Howler(Prey):
    """
    Prey that howls to alert other preys there are predators nearby
    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def update(self, grid: Grid, channels: Dict[str, MessageBroker], sender: MessageSender):
        entities_to_howl_at = []
        predators_to_howl = []

        closest_predator = None
        closest_food = None

        for i in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
            for j in range(-self.nb_tiles_around, self.nb_tiles_around + 1):
                if self.tile[0] + i < 0 or self.tile[0] + i >= grid.tiles.shape[0] or self.tile[1] + j < 0 or \
                        self.tile[1] + j >= grid.tiles.shape[1]:
                    continue

                t = grid.tiles[self.tile[0] + i, self.tile[1] + j]
                for entity in t.get_entities():

                    if isinstance(entity, Predator) or issubclass(type(entity), Predator):
                        predators_to_howl.append(entity.position)
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

                    if isinstance(entity, Prey):
                        entities_to_howl_at.append(entity.id)

        for p in predators_to_howl:
            for e in entities_to_howl_at:
                sender.send(self.id, e, MessageType.PREDATOR_AT, p)

        self.act(closest_predator, closest_food)


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

            self.act(closest_predator, closest_food)


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

        else:
            super().update(map, channels, sender)
