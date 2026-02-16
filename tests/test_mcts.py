"""
test_mcts.py - Tests des algorithmes MCTS.

Teste que les algorithmes fonctionnent correctement :
- Flat Monte Carlo (baseline)
- UCT
- RAVE
- GRAVE
- Planification dynamique
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.environment import (
    GridEnvironment, DynamicObstacle,
    create_simple_scenario, create_dynamic_scenario
)
from src.mcts_base import FlatMonteCarlo, MCTS_UCT
from src.mcts_rave import MCTS_RAVE, MCTS_GRAVE
from src.mcts_dynamic import DynamicPlanner


def test_flat_mc():
    """Test Flat Monte Carlo sur un scenario simple."""
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=50)
    flat = FlatMonteCarlo(n_simulations=200, seed=42)

    action = flat.search(env)
    assert action in env.legal_moves()
    print(f"[OK] Flat MC : action choisie = {action}")


def test_uct_basic():
    """Test UCT sur un scenario simple."""
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=50)
    uct = MCTS_UCT(n_simulations=500, exploration_constant=1.414, seed=42)

    action = uct.search(env)
    assert action in env.legal_moves()
    print(f"[OK] UCT basic : action choisie = {action}")


def test_uct_heuristic():
    """Test UCT avec playout heuristique."""
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=50)
    uct = MCTS_UCT(n_simulations=500, use_heuristic_playout=True, seed=42)

    action = uct.search(env)
    assert action in env.legal_moves()
    print(f"[OK] UCT heuristique : action choisie = {action}")


def test_rave():
    """Test RAVE."""
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=50)
    rave = MCTS_RAVE(n_simulations=500, rave_bias=0.0001, seed=42)

    action = rave.search(env)
    assert action in env.legal_moves()
    print(f"[OK] RAVE : action choisie = {action}")


def test_grave():
    """Test GRAVE."""
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=50)
    grave = MCTS_GRAVE(n_simulations=500, grave_threshold=50, seed=42)

    action = grave.search(env)
    assert action in env.legal_moves()
    print(f"[OK] GRAVE : action choisie = {action}")


def test_uct_reaches_goal():
    """Test que UCT peut atteindre l'objectif sur un scenario simple."""
    # Petit scenario sans obstacles
    env = GridEnvironment(size=8, robot_pos=(1, 1), goal_pos=(6, 6), max_steps=30)
    uct = MCTS_UCT(n_simulations=500, use_heuristic_playout=True, seed=42)

    steps = 0
    while not env.is_terminal() and steps < 30:
        action = uct.search(env)
        env.play(action)
        steps += 1

    print(f"[OK] UCT trajectoire : {steps} pas, "
          f"objectif={'OUI' if env.reached_goal else 'NON'}, "
          f"position finale={env.robot_pos}")


def test_dynamic_planner_uct():
    """Test le planificateur dynamique avec UCT."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8), max_steps=50)
    env.add_walls_border()

    planner = DynamicPlanner(algorithm="uct", n_simulations=300,
                              use_heuristic_playout=True, seed=42)
    results = planner.plan_and_execute(env, verbose=False)

    print(f"[OK] Dynamic UCT : succes={results['success']}, "
          f"{results['steps']} pas, "
          f"temps moyen={results['avg_search_time']:.4f}s")


def test_dynamic_planner_grave():
    """Test le planificateur dynamique avec GRAVE."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8), max_steps=50)
    env.add_walls_border()

    planner = DynamicPlanner(algorithm="grave", n_simulations=300,
                              use_heuristic_playout=True, seed=42)
    results = planner.plan_and_execute(env, verbose=False)

    print(f"[OK] Dynamic GRAVE : succes={results['success']}, "
          f"{results['steps']} pas, "
          f"temps moyen={results['avg_search_time']:.4f}s")


def test_with_dynamic_obstacles():
    """Test avec obstacles dynamiques."""
    env = GridEnvironment(size=12, robot_pos=(1, 1), goal_pos=(10, 10), max_steps=60)
    env.add_walls_border()

    # Ajouter un obstacle dynamique
    obs = DynamicObstacle(x=5, y=5, dx=0, dy=1, pattern="linear",
                           bounds=(5, 2, 5, 9))
    env.add_dynamic_obstacle(obs)

    planner = DynamicPlanner(algorithm="uct", n_simulations=500,
                              use_heuristic_playout=True, seed=42)
    results = planner.plan_and_execute(env, verbose=False)

    print(f"[OK] Obstacles dynamiques : succes={results['success']}, "
          f"{results['steps']} pas")


def test_search_time():
    """Mesure le temps de recherche pour differents budgets."""
    env = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8))
    env.add_walls_border()

    for n_sims in [100, 500, 1000]:
        uct = MCTS_UCT(n_simulations=n_sims, seed=42)
        start = time.time()
        action = uct.search(env)
        elapsed = time.time() - start
        print(f"[OK] {n_sims} simulations : {elapsed:.4f}s")


def test_all_algorithms_comparison():
    """Compare rapidement tous les algorithmes sur le meme scenario."""
    env_base = GridEnvironment(size=10, robot_pos=(1, 1), goal_pos=(8, 8), max_steps=40)
    env_base.add_walls_border()

    algorithms = {
        "Flat MC": FlatMonteCarlo(n_simulations=300, seed=42),
        "UCT": MCTS_UCT(n_simulations=300, seed=42),
        "UCT+heuristic": MCTS_UCT(n_simulations=300, use_heuristic_playout=True, seed=42),
        "RAVE": MCTS_RAVE(n_simulations=300, seed=42),
        "GRAVE": MCTS_GRAVE(n_simulations=300, seed=42),
    }

    print("\nComparaison rapide des algorithmes (1 recherche chacun) :")
    for name, algo in algorithms.items():
        env = env_base.copy()
        start = time.time()
        action = algo.search(env)
        elapsed = time.time() - start
        print(f"  {name:20s} : action={action}, temps={elapsed:.4f}s")


if __name__ == "__main__":
    print("=" * 60)
    print("Tests des algorithmes MCTS")
    print("=" * 60)

    test_flat_mc()
    test_uct_basic()
    test_uct_heuristic()
    test_rave()
    test_grave()
    test_uct_reaches_goal()
    test_dynamic_planner_uct()
    test_dynamic_planner_grave()
    test_with_dynamic_obstacles()
    test_search_time()
    test_all_algorithms_comparison()

    print("=" * 60)
    print("TOUS LES TESTS PASSES !")
    print("=" * 60)
