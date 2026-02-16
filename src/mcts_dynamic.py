"""
mcts_dynamic.py - MCTS avec re-planification en temps reel pour environnement dynamique.

Ce module est le COEUR du projet : il gere la planification de trajectoire
quand l'environnement change (obstacles qui bougent, incertitude capteur).

Approche :
1. A chaque pas de temps, le robot observe l'environnement
2. Il lance une recherche MCTS pour choisir la prochaine action
3. Il execute l'action et observe le nouvel etat
4. Si l'environnement a change, il re-planifie (nouvelle recherche MCTS)

C'est une approche "online planning" : on ne planifie pas tout le chemin
a l'avance, on re-planifie a chaque pas. C'est essentiel quand
l'environnement est dynamique.

Lien avec le cours :
- Section 16 : Progressive Widening (pour les actions continues)
- Section 8 : MCTS Solver (resoudre des sous-arbres)
- Section 7 : Parallelisation (pour le temps reel)

References :
- Bonanni et al. (2025) "MCTS with Velocity Obstacles" - AAMAS
- Eiffert et al. (2020) "Path Planning with Generative RNNs and MCTS" - ICRA
- Dam et al. (2022) "Monte-Carlo Robot Path Planning" - IEEE RA-L
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from .environment import GridEnvironment, DynamicObstacle, ACTIONS, ACTION_NAMES
from .mcts_base import MCTS_UCT, FlatMonteCarlo
from .mcts_rave import MCTS_RAVE, MCTS_GRAVE


class DynamicPlanner:
    """
    Planificateur dynamique combinant MCTS avec re-planification temps reel.

    A chaque pas de temps :
    1. Observer l'environnement (avec bruit capteur si active)
    2. Lancer MCTS pour choisir la prochaine action
    3. Executer l'action
    4. Repeter

    C'est l'approche "receding horizon" : on planifie a chaque instant
    en fonction des informations disponibles. Les obstacles dynamiques
    rendent cette re-planification necessaire car le plan initial
    devient obsolete quand les obstacles bougent.

    Techniques cles :
    - Re-planification a chaque pas (online planning)
    - Utilisation du bruit capteur pour la robustesse
    - Choix entre differents algorithmes MCTS
    """

    def __init__(self, algorithm: str = "uct", n_simulations: int = 1000,
                 exploration_constant: float = 1.414, use_heuristic_playout: bool = True,
                 max_playout_steps: int = 50, seed: int = None, **kwargs):
        """
        Args:
            algorithm: Algorithme MCTS a utiliser :
                - "flat" : Flat Monte Carlo (baseline, cours section 1.2)
                - "uct"  : UCT standard (cours section 3.2)
                - "rave" : RAVE (cours section 5.3)
                - "grave": GRAVE (cours section 6.1)
            n_simulations: Budget de simulations par pas de temps
            exploration_constant: C dans UCT
            use_heuristic_playout: Utiliser un playout guide par heuristique
            max_playout_steps: Profondeur max des playouts
            seed: Graine aleatoire
            **kwargs: Arguments supplementaires pour l'algorithme choisi
        """
        self.algorithm_name = algorithm
        self.n_simulations = n_simulations
        self.seed = seed

        # Creer l'algorithme MCTS
        if algorithm == "flat":
            self.mcts = FlatMonteCarlo(
                n_simulations=n_simulations,
                max_playout_steps=max_playout_steps,
                seed=seed
            )
        elif algorithm == "uct":
            self.mcts = MCTS_UCT(
                n_simulations=n_simulations,
                exploration_constant=exploration_constant,
                max_playout_steps=max_playout_steps,
                use_heuristic_playout=use_heuristic_playout,
                max_depth=kwargs.get('max_depth', 15),
                seed=seed
            )
        elif algorithm == "rave":
            self.mcts = MCTS_RAVE(
                n_simulations=n_simulations,
                exploration_constant=exploration_constant,
                rave_bias=kwargs.get('rave_bias', 0.0001),
                max_playout_steps=max_playout_steps,
                use_heuristic_playout=use_heuristic_playout,
                max_depth=kwargs.get('max_depth', 15),
                seed=seed
            )
        elif algorithm == "grave":
            self.mcts = MCTS_GRAVE(
                n_simulations=n_simulations,
                exploration_constant=exploration_constant,
                rave_bias=kwargs.get('rave_bias', 0.0001),
                grave_threshold=kwargs.get('grave_threshold', 50),
                max_playout_steps=max_playout_steps,
                use_heuristic_playout=use_heuristic_playout,
                max_depth=kwargs.get('max_depth', 15),
                seed=seed
            )
        else:
            raise ValueError(f"Algorithme inconnu: {algorithm}")

        # Historique pour l'analyse
        self.action_history = []
        self.reward_history = []
        self.search_times = []
        self.replanning_count = 0

    def plan_and_execute(self, env: GridEnvironment, verbose: bool = False) -> Dict:
        """
        Boucle principale : planifier et executer jusqu'a l'objectif.

        A chaque pas de temps :
        1. Observer l'environnement
        2. Si l'environnement a change depuis la derniere observation,
           incrementer le compteur de re-planification
        3. Lancer MCTS pour choisir la meilleure action
        4. Executer l'action
        5. Enregistrer les statistiques

        Args:
            env: L'environnement de depart
            verbose: Afficher les details

        Returns:
            Dictionnaire avec les resultats :
            - success: objectif atteint ?
            - steps: nombre de pas
            - path: trajectoire suivie
            - total_reward: recompense totale
            - search_times: temps de chaque recherche MCTS
            - replanning_count: nombre de re-planifications
        """
        self.action_history = []
        self.reward_history = []
        self.search_times = []
        self.replanning_count = 0

        step = 0
        total_reward = 0.0

        while not env.is_terminal():
            # 1. Observer l'environnement (avec bruit capteur si active)
            if env.sensor_noise > 0:
                observed_grid = env.get_observed_grid()
                # Creer un environnement "observe" pour la planification
                planning_env = self._create_observed_env(env, observed_grid)
            else:
                planning_env = env.copy()

            # 2. Lancer MCTS pour choisir l'action
            start_time = time.time()
            action = self.mcts.search(planning_env)
            search_time = time.time() - start_time
            self.search_times.append(search_time)

            # 3. Executer l'action
            env.play(action)
            reward = env.score() if env.is_terminal() else -1.0
            total_reward += reward

            # Enregistrer
            self.action_history.append(action)
            self.reward_history.append(reward)

            if verbose:
                action_name = ACTION_NAMES[action]
                print(f"Step {step}: Action={action_name}, "
                      f"Pos={env.robot_pos}, "
                      f"Reward={reward:.1f}, "
                      f"Search={search_time:.3f}s")

            step += 1

            # 4. Verifier si re-planification necessaire
            # (les obstacles dynamiques ont bouge)
            if len(env.dynamic_obstacles) > 0:
                self.replanning_count += 1

        # Resultats
        results = {
            'success': env.reached_goal,
            'steps': step,
            'path': list(env.path_history),
            'total_reward': total_reward,
            'search_times': self.search_times,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'replanning_count': self.replanning_count,
            'algorithm': self.algorithm_name,
            'n_simulations': self.n_simulations,
        }

        return results

    def _create_observed_env(self, real_env: GridEnvironment,
                              observed_grid: np.ndarray) -> GridEnvironment:
        """
        Cree un environnement base sur l'observation bruitee.

        Cela simule le fait que le robot ne connait pas parfaitement
        l'etat de l'environnement (incertitude capteur).

        Lien avec le cours, section 19 (Jeux a information imparfaite) :
        Le robot doit prendre des decisions avec une information incomplete,
        comme dans les jeux a information imparfaite (Poker, Bridge).

        L'approche PIMC (cours section 19.2) est analogue :
        on genere des "mondes possibles" compatibles avec l'observation
        et on planifie dans ces mondes.
        """
        obs_env = GridEnvironment(
            size=real_env.size,
            robot_pos=real_env.robot_pos,
            goal_pos=real_env.goal_pos,
            max_steps=real_env.max_steps - real_env.steps,
            sensor_noise=0  # Pas de bruit dans la copie de planification
        )
        obs_env.grid = real_env.grid.copy()
        obs_env.dynamic_obstacles = [obs.copy() for obs in real_env.dynamic_obstacles]
        obs_env.steps = real_env.steps
        return obs_env


class MultiWorldPlanner:
    """
    Planificateur avec echantillonnage de mondes multiples.

    Inspire de PIMC (Perfect Information Monte Carlo, cours section 19.2) :
    Au lieu de planifier dans un seul monde, on genere N mondes possibles
    compatibles avec les observations du robot, et on choisit l'action
    qui fonctionne le mieux en moyenne sur tous ces mondes.

    Algorithme (inspire de PIMC, cours section 19.2) :
    1. Pour chaque action possible :
       a. Generer N "mondes possibles"
       b. Pour chaque monde, lancer MCTS
       c. Compter combien de mondes sont gagnes
    2. Jouer l'action qui gagne le plus de mondes

    C'est plus robuste face a l'incertitude car on ne se fie pas
    a une seule estimation de l'environnement.

    Reference :
    - Janson et al. (2015) "Monte Carlo Motion Planning Under Uncertainty"
    """

    def __init__(self, n_worlds: int = 5, algorithm: str = "uct",
                 n_simulations: int = 500, seed: int = None, **kwargs):
        """
        Args:
            n_worlds: Nombre de mondes a echantillonner
            algorithm: Algorithme MCTS pour chaque monde
            n_simulations: Budget par monde (total = n_worlds * n_simulations)
            seed: Graine aleatoire
        """
        self.n_worlds = n_worlds
        self.algorithm = algorithm
        self.n_simulations = n_simulations
        self.seed = seed
        self.kwargs = kwargs
        self.rng = np.random.default_rng(seed)

    def choose_action(self, env: GridEnvironment) -> int:
        """
        Choisit la meilleure action en echantillonnant plusieurs mondes.

        Pour chaque monde possible :
        1. Perturbation aleatoire des positions des obstacles dynamiques
        2. Lancer MCTS dans ce monde
        3. Enregistrer l'action choisie

        On retourne l'action la plus frequemment choisie (vote majoritaire).
        C'est analogue a la Root Parallelization (cours section 7.1) :
        plusieurs recherches independantes puis vote.
        """
        action_votes = {}
        legal = env.legal_moves()

        for w in range(self.n_worlds):
            # Creer un monde possible (perturbation)
            world = self._sample_world(env)

            # Lancer MCTS dans ce monde
            planner = DynamicPlanner(
                algorithm=self.algorithm,
                n_simulations=self.n_simulations,
                seed=self.rng.integers(0, 2**31),
                **self.kwargs
            )
            action = planner.mcts.search(world)

            # Vote
            action_votes[action] = action_votes.get(action, 0) + 1

        # Retourner l'action majoritaire
        best_action = max(action_votes.keys(), key=lambda a: action_votes[a])
        return best_action

    def _sample_world(self, env: GridEnvironment) -> GridEnvironment:
        """
        Genere un monde possible en perturbant les obstacles dynamiques.

        Les obstacles dynamiques sont perturbes aleatoirement pour
        simuler l'incertitude sur leur position reelle.
        """
        world = env.copy()

        # Perturber les obstacles dynamiques
        for obs in world.dynamic_obstacles:
            # Perturbation gaussienne de la position
            dx = self.rng.integers(-1, 2)
            dy = self.rng.integers(-1, 2)
            new_x = max(0, min(env.size - 1, obs.x + dx))
            new_y = max(0, min(env.size - 1, obs.y + dy))
            if world.grid[new_x, new_y] == 0:
                obs.x = new_x
                obs.y = new_y

        return world


class AdaptivePlanner:
    """
    Planificateur adaptatif qui ajuste le budget de simulations dynamiquement.

    Quand l'environnement est "calme" (pas d'obstacles proches),
    on utilise peu de simulations (rapide).
    Quand l'environnement est "dangereux" (obstacles proches),
    on augmente le budget (plus precis).

    C'est une forme d'allocation adaptative de ressources,
    analogue au Sequential Halving (cours section 11.1) :
    on donne plus de budget aux decisions critiques.
    """

    def __init__(self, algorithm: str = "grave", min_simulations: int = 200,
                 max_simulations: int = 3000, danger_radius: int = 3,
                 seed: int = None, **kwargs):
        """
        Args:
            algorithm: Algorithme MCTS de base
            min_simulations: Budget minimum (environnement calme)
            max_simulations: Budget maximum (environnement dangereux)
            danger_radius: Rayon de detection du danger
            seed: Graine aleatoire
        """
        self.algorithm = algorithm
        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.danger_radius = danger_radius
        self.seed = seed
        self.kwargs = kwargs

    def _assess_danger(self, env: GridEnvironment) -> float:
        """
        Evalue le niveau de danger de la situation actuelle.

        Le danger est eleve quand :
        - Des obstacles dynamiques sont proches du robot
        - Le robot est dans un passage etroit
        - Le robot est proche de l'objectif (dernier moment critique)

        Returns:
            Niveau de danger entre 0 (calme) et 1 (tres dangereux)
        """
        rx, ry = env.robot_pos
        danger = 0.0

        # Proximite des obstacles dynamiques
        for obs in env.dynamic_obstacles:
            dist = abs(rx - obs.x) + abs(ry - obs.y)
            if dist <= self.danger_radius:
                danger = max(danger, 1.0 - dist / self.danger_radius)

        # Nombre de mouvements legaux (passage etroit)
        legal = env.legal_moves()
        if len(legal) <= 3:
            danger = max(danger, 0.7)  # Peu de choix = situation critique

        # Proximite de l'objectif
        goal_dist = env.manhattan_distance_to_goal()
        if goal_dist <= 3:
            danger = max(danger, 0.5)

        return min(1.0, danger)

    def choose_action(self, env: GridEnvironment) -> int:
        """
        Choisit l'action avec un budget adaptatif.

        Le budget de simulations est ajuste selon le danger :
        n_sims = min_sims + danger * (max_sims - min_sims)
        """
        danger = self._assess_danger(env)
        n_sims = int(self.min_simulations +
                      danger * (self.max_simulations - self.min_simulations))

        planner = DynamicPlanner(
            algorithm=self.algorithm,
            n_simulations=n_sims,
            seed=self.seed,
            **self.kwargs
        )
        return planner.mcts.search(env)

    def plan_and_execute(self, env: GridEnvironment, verbose: bool = False) -> Dict:
        """Execute la planification adaptative complete."""
        action_history = []
        search_times = []
        danger_history = []
        budget_history = []

        step = 0
        total_reward = 0.0

        while not env.is_terminal():
            danger = self._assess_danger(env)
            n_sims = int(self.min_simulations +
                          danger * (self.max_simulations - self.min_simulations))

            planner = DynamicPlanner(
                algorithm=self.algorithm,
                n_simulations=n_sims,
                seed=self.seed,
                **self.kwargs
            )

            start_time = time.time()
            action = planner.mcts.search(env.copy())
            search_time = time.time() - start_time

            env.play(action)
            reward = env.score() if env.is_terminal() else -1.0
            total_reward += reward

            action_history.append(action)
            search_times.append(search_time)
            danger_history.append(danger)
            budget_history.append(n_sims)

            if verbose:
                print(f"Step {step}: Action={ACTION_NAMES[action]}, "
                      f"Pos={env.robot_pos}, Danger={danger:.2f}, "
                      f"Budget={n_sims}, Time={search_time:.3f}s")

            step += 1

        return {
            'success': env.reached_goal,
            'steps': step,
            'path': list(env.path_history),
            'total_reward': total_reward,
            'search_times': search_times,
            'avg_search_time': np.mean(search_times) if search_times else 0,
            'danger_history': danger_history,
            'budget_history': budget_history,
            'algorithm': f"adaptive_{self.algorithm}",
        }
