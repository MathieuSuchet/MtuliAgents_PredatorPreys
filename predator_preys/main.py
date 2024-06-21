import pygame.event

from predator_preys.managers import Manager
from predator_preys.utils import GameConfig, PreyConfig, PredatorConfig, FoodConfig

n_predators = 5
n_preys = 5
n_food = 5

game_config = GameConfig(
    PreyConfig(
        nb_preys=n_preys,
        decoys_percentage=0.1,
        runner_percentage=0.8,
        howler_percentage=0.1
    ),
    PredatorConfig(
        nb_predators=n_predators,
        leader_percentage=0.1,
        follower_percentage=0.9
    ),
    FoodConfig(
        nb_food=n_food
    )
)

m = Manager(game_config)
running = True

while running:
    m.render()
    m.step()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            running = False

    if (len(m.map.entities.preys) == 0
            or len(m.map.entities.predators) == 0
            or len(m.map.entities.preys) > 300
            or len(m.map.entities.predators) > 300):
        m.reset()
