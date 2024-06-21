from typing import List

from predator_preys.utils import Predator, Food, Prey


class EntityBuffer:
    def __init__(self):
        self.foods: List[Food] = []
        self.predators: List[Predator] = []
        self.preys: List[Prey] = []