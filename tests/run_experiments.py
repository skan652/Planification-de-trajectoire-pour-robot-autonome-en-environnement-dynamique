"""
run_experiments.py - Script principal d'experimentations.

Ce script lance les experimentations systematiques pour comparer
les differents algorithmes MCTS sur les differents scenarios.

Experimentations :
1. Comparaison Flat MC vs UCT vs RAVE vs GRAVE (scenario statique)
2. Impact du nombre de simulations
3. Impact de l'exploration constant C
4. Performance sur environnement dynamique
5. Robustesse au bruit capteur
6. Planification adaptative

Chaque experimentation est repetee N fois pour obtenir des statistiques
fiables (cf. cours section 25.3, erreur #5 : tester avec suffisamment de parties).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from tqdm import tqdm
from src.environment import (
    GridEnvironment, DynamicObstacle,
    create_simple_scenario, create_dynamic_scenario,
    create_complex_scenario, create_narrow_passage_scenario,
    create_sensor_noise_scenario
)
from src.mcts_base import FlatMonteCarlo, MCTS_UCT
from src.mcts_rave import MCTS_RAVE, MCTS_GRAVE
from src.mcts_dynamic import DynamicPlanner, AdaptivePlanner
from src.visualization import (
    plot_environment, plot_multi_comparison,
    plot_trajectory_comparison, plot_search_time_evolution,
    plot_comparison
)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_experiment(env_factory, planner_factory, n_runs: int = 20,
                           verbose: bool = False) -> list:
    """
    Lance n_runs executions d'un planificateur sur un environnement.

    Args:
        env_factory: Fonction qui cree un nouvel environnement
        planner_factory: Fonction qui cree un nouveau planificateur
        n_runs: Nombre de repetitions
        verbose: Afficher les details

    Returns:
        Liste de dictionnaires de resultats
    """
    results = []
    for i in range(n_runs):
        env = env_factory()
        planner = planner_factory(seed=i*42)
        result = planner.plan_and_execute(env, verbose=verbose)
        results.append(result)

        if verbose:
            status = "SUCCES" if result['success'] else "ECHEC"
            print(f"  Run {i+1}/{n_runs}: {status}, {result['steps']} pas")

    return results


def experiment_1_algorithm_comparison():
    """
    Experimentation 1 : Comparaison des algorithmes sur scenario statique.

    Compare : Flat MC, UCT, UCT+heuristique, RAVE, GRAVE

    Objectif : Montrer que les ameliorations vues en cours
    (UCT > Flat MC, RAVE > UCT, GRAVE > RAVE) se verifient
    dans notre contexte de planification de trajectoire.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 1 : Comparaison des algorithmes (scenario statique)")
    print("=" * 70)

    N_RUNS = 15
    N_SIMS = 500

    def env_factory():
        return create_simple_scenario(15)

    algorithms = {
        "Flat MC": lambda seed: DynamicPlanner("flat", n_simulations=N_SIMS, seed=seed),
        "UCT": lambda seed: DynamicPlanner("uct", n_simulations=N_SIMS, seed=seed),
        "UCT+heur.": lambda seed: DynamicPlanner("uct", n_simulations=N_SIMS,
                                                    use_heuristic_playout=True, seed=seed),
        "RAVE": lambda seed: DynamicPlanner("rave", n_simulations=N_SIMS,
                                             use_heuristic_playout=True, seed=seed),
        "GRAVE": lambda seed: DynamicPlanner("grave", n_simulations=N_SIMS,
                                              use_heuristic_playout=True, seed=seed),
    }

    all_results = {}
    for name, planner_factory in algorithms.items():
        print(f"\n--- {name} ---")
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_steps = np.mean([r['steps'] for r in results if r['success']]) if any(r['success'] for r in results) else float('inf')
        avg_time = np.mean([r['avg_search_time'] for r in results])

        print(f"  Taux de succes : {success_rate:.1%}")
        print(f"  Pas moyens (succes) : {avg_steps:.1f}")
        print(f"  Temps moyen/recherche : {avg_time:.4f}s")

    # Visualisation
    plot_multi_comparison(all_results,
                           save_path=os.path.join(RESULTS_DIR, "exp1_comparison.png"),
                           show=False)
    print("\n[Graphique sauvegarde : results/exp1_comparison.png]")

    return all_results


def experiment_2_simulation_budget():
    """
    Experimentation 2 : Impact du nombre de simulations.

    Teste UCT avec differents budgets : 100, 300, 500, 1000, 2000

    Objectif : Montrer que plus de simulations = meilleure performance,
    mais avec des rendements decroissants (cf. cours, le budget est
    typiquement 1000-10000).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 2 : Impact du budget de simulations")
    print("=" * 70)

    N_RUNS = 10
    budgets = [100, 300, 500, 1000, 2000]

    def env_factory():
        return create_simple_scenario(15)

    all_results = {}
    for budget in budgets:
        name = f"UCT-{budget}"
        print(f"\n--- {name} ---")
        planner_factory = lambda seed, b=budget: DynamicPlanner(
            "uct", n_simulations=b, use_heuristic_playout=True, seed=seed
        )
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_time = np.mean([r['avg_search_time'] for r in results])
        print(f"  Taux de succes : {success_rate:.1%}, Temps : {avg_time:.4f}s")

    plot_comparison(all_results, metric='success_rate',
                     title='Taux de succes vs Budget de simulations',
                     save_path=os.path.join(RESULTS_DIR, "exp2_budget.png"),
                     show=False)
    print("\n[Graphique sauvegarde : results/exp2_budget.png]")

    return all_results


def experiment_3_exploration_constant():
    """
    Experimentation 3 : Impact de la constante d'exploration C.

    Teste UCT avec C = 0.5, 1.0, 1.414, 2.0, 3.0

    Objectif : Montrer l'equilibre exploration/exploitation (cours section 2).
    - C trop petit = trop d'exploitation, peut rester bloque
    - C trop grand = trop d'exploration, perd du temps
    - C optimal = sqrt(2) ~= 1.414 (recommandation du cours)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 3 : Impact de la constante d'exploration C")
    print("=" * 70)

    N_RUNS = 10
    c_values = [0.5, 1.0, 1.414, 2.0, 3.0]

    def env_factory():
        return create_simple_scenario(15)

    all_results = {}
    for c in c_values:
        name = f"C={c}"
        print(f"\n--- {name} ---")
        planner_factory = lambda seed, c_val=c: DynamicPlanner(
            "uct", n_simulations=500, exploration_constant=c_val,
            use_heuristic_playout=True, seed=seed
        )
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  Taux de succes : {success_rate:.1%}")

    plot_comparison(all_results, metric='success_rate',
                     title='Taux de succes vs Constante C',
                     save_path=os.path.join(RESULTS_DIR, "exp3_exploration.png"),
                     show=False)
    print("\n[Graphique sauvegarde : results/exp3_exploration.png]")

    return all_results


def experiment_4_dynamic_environment():
    """
    Experimentation 4 : Performance en environnement dynamique.

    Compare les algorithmes sur le scenario avec obstacles dynamiques.

    Objectif : Montrer que la re-planification est necessaire
    et que GRAVE gere mieux les environnements dynamiques.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 4 : Environnement dynamique")
    print("=" * 70)

    N_RUNS = 15

    def env_factory():
        return create_dynamic_scenario(15)

    algorithms = {
        "UCT": lambda seed: DynamicPlanner("uct", n_simulations=500,
                                            use_heuristic_playout=True, seed=seed),
        "RAVE": lambda seed: DynamicPlanner("rave", n_simulations=500,
                                             use_heuristic_playout=True, seed=seed),
        "GRAVE": lambda seed: DynamicPlanner("grave", n_simulations=500,
                                              use_heuristic_playout=True, seed=seed),
    }

    all_results = {}
    for name, planner_factory in algorithms.items():
        print(f"\n--- {name} ---")
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  Taux de succes : {success_rate:.1%}")

    plot_multi_comparison(all_results,
                           save_path=os.path.join(RESULTS_DIR, "exp4_dynamic.png"),
                           show=False)
    print("\n[Graphique sauvegarde : results/exp4_dynamic.png]")

    return all_results


def experiment_5_sensor_noise():
    """
    Experimentation 5 : Robustesse au bruit capteur.

    Teste GRAVE avec differents niveaux de bruit : 0%, 5%, 10%, 20%, 30%

    Objectif : Montrer l'impact de l'incertitude d'observation
    sur la performance de planification.
    Lien avec le cours section 19 (jeux a information imparfaite).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 5 : Impact du bruit capteur")
    print("=" * 70)

    N_RUNS = 10
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

    all_results = {}
    for noise in noise_levels:
        name = f"Bruit={noise:.0%}"
        print(f"\n--- {name} ---")

        def env_factory(n=noise):
            return create_sensor_noise_scenario(15, noise=n)

        planner_factory = lambda seed: DynamicPlanner(
            "grave", n_simulations=500, use_heuristic_playout=True, seed=seed
        )
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  Taux de succes : {success_rate:.1%}")

    plot_comparison(all_results, metric='success_rate',
                     title='Taux de succes vs Niveau de bruit capteur',
                     save_path=os.path.join(RESULTS_DIR, "exp5_noise.png"),
                     show=False)
    print("\n[Graphique sauvegarde : results/exp5_noise.png]")

    return all_results


def experiment_6_narrow_passage():
    """
    Experimentation 6 : Scenario de passage etroit.

    Teste la capacite des algorithmes a trouver et traverser
    un passage etroit avec un obstacle dynamique.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTATION 6 : Passage etroit avec obstacle dynamique")
    print("=" * 70)

    N_RUNS = 10

    def env_factory():
        return create_narrow_passage_scenario(15)

    algorithms = {
        "UCT": lambda seed: DynamicPlanner("uct", n_simulations=800,
                                            use_heuristic_playout=True, seed=seed),
        "GRAVE": lambda seed: DynamicPlanner("grave", n_simulations=800,
                                              use_heuristic_playout=True, seed=seed),
    }

    all_results = {}
    for name, planner_factory in algorithms.items():
        print(f"\n--- {name} ---")
        results = run_single_experiment(env_factory, planner_factory, N_RUNS)
        all_results[name] = results

        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  Taux de succes : {success_rate:.1%}")

    plot_multi_comparison(all_results,
                           save_path=os.path.join(RESULTS_DIR, "exp6_narrow.png"),
                           show=False)
    print("\n[Graphique sauvegarde : results/exp6_narrow.png]")

    return all_results


def generate_trajectory_visualization():
    """
    Genere une visualisation comparative des trajectoires.
    """
    print("\n" + "=" * 70)
    print("GENERATION DES VISUALISATIONS DE TRAJECTOIRES")
    print("=" * 70)

    env_base = create_simple_scenario(15)

    # Visualiser l'environnement initial
    plot_environment(env_base, title="Environnement initial",
                      save_path=os.path.join(RESULTS_DIR, "environment_initial.png"),
                      show=False)

    # Generer une trajectoire par algorithme
    trajectories = {}
    for algo in ["uct", "grave"]:
        env = create_simple_scenario(15)
        planner = DynamicPlanner(algo, n_simulations=500,
                                  use_heuristic_playout=True, seed=42)
        result = planner.plan_and_execute(env, verbose=False)
        trajectories[algo.upper()] = result['path']
        print(f"  {algo.upper()}: {result['steps']} pas, succes={result['success']}")

    plot_trajectory_comparison(env_base, trajectories,
                                save_path=os.path.join(RESULTS_DIR, "trajectories.png"),
                                show=False)
    print("\n[Graphiques sauvegardes dans results/]")


def run_all_experiments():
    """Lance toutes les experimentations."""
    print("\n" + "#" * 70)
    print("#  EXPERIMENTATIONS MCTS POUR PLANIFICATION DE TRAJECTOIRE ROBOT")
    print("#" * 70)

    start = time.time()

    exp1 = experiment_1_algorithm_comparison()
    exp2 = experiment_2_simulation_budget()
    exp3 = experiment_3_exploration_constant()
    exp4 = experiment_4_dynamic_environment()
    exp5 = experiment_5_sensor_noise()
    exp6 = experiment_6_narrow_passage()
    generate_trajectory_visualization()

    total_time = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"TOUTES LES EXPERIMENTATIONS TERMINEES en {total_time:.1f}s")
    print(f"Resultats sauvegardes dans : {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # On peut lancer toutes les experimentations ou une seule
    import argparse
    parser = argparse.ArgumentParser(description="Experimentations MCTS")
    parser.add_argument("--exp", type=int, default=0,
                         help="Numero d'experimentation (0=toutes, 1-6=specifique)")
    parser.add_argument("--viz", action="store_true",
                         help="Generer les visualisations uniquement")
    args = parser.parse_args()

    if args.viz:
        generate_trajectory_visualization()
    elif args.exp == 0:
        run_all_experiments()
    elif args.exp == 1:
        experiment_1_algorithm_comparison()
    elif args.exp == 2:
        experiment_2_simulation_budget()
    elif args.exp == 3:
        experiment_3_exploration_constant()
    elif args.exp == 4:
        experiment_4_dynamic_environment()
    elif args.exp == 5:
        experiment_5_sensor_noise()
    elif args.exp == 6:
        experiment_6_narrow_passage()
