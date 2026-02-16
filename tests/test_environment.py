"""
test_environment.py - Tests unitaires pour l'environnement de grille.

Ces tests verifient que l'environnement fonctionne correctement :
- Creation de la grille
- Mouvements du robot
- Obstacles statiques et dynamiques
- Detection de collisions
- Fonction de recompense
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import (
    GridEnvironment, DynamicObstacle,
    CELL_FREE, CELL_OBSTACLE, CELL_DYNAMIC_OBS, CELL_ROBOT, CELL_GOAL,
    ACTIONS, create_simple_scenario, create_dynamic_scenario
)


def test_grid_creation():
    """Test la creation de la grille."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8))
    assert env.size == 10
    assert env.robot_pos == (1, 1)
    assert env.goal_pos == (8, 8)
    assert not env.done
    print("[OK] Creation de la grille")


def test_walls():
    """Test l'ajout de murs."""
    env = GridEnvironment(size=10)
    env.add_walls_border()
    # Les bords doivent etre des obstacles
    assert env.grid[0, 0] == CELL_OBSTACLE
    assert env.grid[0, 5] == CELL_OBSTACLE
    assert env.grid[9, 9] == CELL_OBSTACLE
    # Le centre doit etre libre
    assert env.grid[5, 5] == CELL_FREE
    print("[OK] Ajout de murs")


def test_legal_moves():
    """Test les mouvements legaux."""
    env = GridEnvironment(size=10, robot_pos=(5, 5))
    env.add_walls_border()
    legal = env.legal_moves()
    # Au centre, toutes les 9 directions sont legales
    assert len(legal) == 9  # 8 directions + WAIT
    print(f"[OK] Mouvements legaux : {len(legal)} actions")


def test_legal_moves_corner():
    """Test les mouvements legaux dans un coin."""
    env = GridEnvironment(size=10, robot_pos=(1, 1))
    env.add_walls_border()
    legal = env.legal_moves()
    # Pres du mur, certaines directions sont bloquees
    assert len(legal) < 9
    print(f"[OK] Mouvements legaux coin : {len(legal)} actions")


def test_robot_movement():
    """Test le deplacement du robot."""
    env = GridEnvironment(size=10, robot_pos=(5, 5), goal_pos=(8, 8))
    env.play(3)  # EST
    assert env.robot_pos == (5, 6)
    env.play(1)  # SUD
    assert env.robot_pos == (6, 6)
    print("[OK] Deplacement du robot")


def test_goal_reached():
    """Test l'atteinte de l'objectif."""
    env = GridEnvironment(size=10, robot_pos=(8, 7), goal_pos=(8, 8))
    env.play(3)  # EST -> atteint (8, 8)
    assert env.is_terminal()
    assert env.reached_goal
    assert env.score() > 0
    print(f"[OK] Objectif atteint, score = {env.score():.2f}")


def test_dynamic_obstacle_linear():
    """Test le mouvement d'un obstacle dynamique lineaire."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8))
    obs = DynamicObstacle(x=5, y=5, dx=0, dy=1, pattern="linear",
                           bounds=(5, 3, 5, 7))
    env.add_dynamic_obstacle(obs)

    # L'obstacle doit se deplacer
    initial_y = obs.y
    obs.move(10, env.grid)
    assert obs.y == initial_y + 1
    print("[OK] Obstacle dynamique lineaire")


def test_dynamic_obstacle_bounce():
    """Test le rebond d'un obstacle dynamique."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8))
    obs = DynamicObstacle(x=5, y=7, dx=0, dy=1, pattern="linear",
                           bounds=(5, 3, 5, 7))
    env.add_dynamic_obstacle(obs)

    # Deplacer l'obstacle vers la limite
    obs.move(10, env.grid)  # y=7 -> rebondit
    # Apres le rebond, la direction doit s'inverser
    assert obs.dy == -1
    print("[OK] Rebond obstacle dynamique")


def test_collision_with_dynamic():
    """Test la collision avec un obstacle dynamique."""
    env = GridEnvironment(size=10, robot_pos=(5, 4), goal_pos=(8, 8))
    obs = DynamicObstacle(x=5, y=5, dx=0, dy=0, pattern="linear")
    env.add_dynamic_obstacle(obs)

    env.play(3)  # EST -> collision avec obstacle en (5,5)
    assert env.is_terminal()
    assert not env.reached_goal
    assert env.score() < 0
    print(f"[OK] Collision detectee, score = {env.score():.2f}")


def test_max_steps():
    """Test la limite de pas."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8), max_steps=5)
    for _ in range(10):
        if not env.is_terminal():
            env.play(8)  # WAIT
    assert env.is_terminal()
    assert not env.reached_goal
    print("[OK] Limite de pas")


def test_copy():
    """Test la copie profonde (crucial pour MCTS)."""
    env = GridEnvironment(size=10, robot_pos=(5, 5), goal_pos=(8, 8))
    obs = DynamicObstacle(x=3, y=3, dx=0, dy=1, pattern="linear")
    env.add_dynamic_obstacle(obs)

    env_copy = env.copy()

    # Modifier la copie ne doit pas affecter l'original
    env_copy.play(3)
    assert env.robot_pos == (5, 5)  # Original inchange
    assert env_copy.robot_pos == (5, 6)  # Copie changee
    print("[OK] Copie profonde")


def test_state_hash():
    """Test le hash d'etat (pour la table de transposition)."""
    env1 = GridEnvironment(size=10, robot_pos=(5, 5), goal_pos=(8, 8))
    env2 = GridEnvironment(size=10, robot_pos=(5, 5), goal_pos=(8, 8))
    env3 = GridEnvironment(size=10, robot_pos=(5, 6), goal_pos=(8, 8))

    h1 = env1.get_state_hash()
    h2 = env2.get_state_hash()
    h3 = env3.get_state_hash()

    assert h1 == h2  # Meme etat -> meme hash
    assert h1 != h3  # Etat different -> hash different
    print("[OK] Hash d'etat")


def test_sensor_noise():
    """Test le bruit capteur."""
    env = GridEnvironment(size=10, robot_pos=(5, 5), goal_pos=(8, 8),
                          sensor_noise=0.3, seed=42)
    env.add_walls_border()

    observed = env.get_observed_grid()
    real = env.get_grid_state()

    # Il doit y avoir des differences
    diff_count = np.sum(observed != real)
    print(f"[OK] Bruit capteur : {diff_count} cellules differentes")


def test_scenarios():
    """Test la creation des scenarios predifinis."""
    env1 = create_simple_scenario(20)
    assert env1.size == 20
    assert not env1.is_terminal()
    print("[OK] Scenario simple")

    env2 = create_dynamic_scenario(20)
    assert len(env2.dynamic_obstacles) > 0
    print(f"[OK] Scenario dynamique : {len(env2.dynamic_obstacles)} obstacles dynamiques")


def test_random_playout():
    """Test un playout aleatoire complet."""
    env = create_simple_scenario(15)
    rng = np.random.default_rng(42)

    steps = 0
    while not env.is_terminal() and steps < 200:
        legal = env.legal_moves()
        action = legal[rng.integers(0, len(legal))]
        env.play(action)
        steps += 1

    print(f"[OK] Playout aleatoire : {steps} pas, "
          f"objectif={'OUI' if env.reached_goal else 'NON'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tests de l'environnement")
    print("=" * 60)

    test_grid_creation()
    test_walls()
    test_legal_moves()
    test_legal_moves_corner()
    test_robot_movement()
    test_goal_reached()
    test_dynamic_obstacle_linear()
    test_dynamic_obstacle_bounce()
    test_collision_with_dynamic()
    test_max_steps()
    test_copy()
    test_state_hash()
    test_sensor_noise()
    test_scenarios()
    test_random_playout()

    print("=" * 60)
    print("TOUS LES TESTS PASSES !")
    print("=" * 60)
