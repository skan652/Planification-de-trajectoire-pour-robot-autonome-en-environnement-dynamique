"""
environment.py - Environnement de grille 2D pour la planification de trajectoire robot.

Ce module implemente l'environnement dans lequel le robot evolue :
- Grille 2D avec obstacles statiques et dynamiques
- Modele de mouvement des obstacles dynamiques
- Detection de collisions
- Fonction de recompense

Lien avec le cours :
- L'environnement definit les etats, actions et transitions du MDP
- Les obstacles dynamiques introduisent l'aspect stochastique
- La fonction de recompense guide la recherche MCTS

References :
- Dam et al. (2022) "Monte-Carlo Robot Path Planning" - IEEE RA-L
- Bonanni et al. (2025) "MCTS with Velocity Obstacles" - AAMAS 2025
"""

import numpy as np
import copy
from typing import List, Tuple, Optional


# ============================================================================
# CELLULES DE LA GRILLE
# ============================================================================
# Chaque cellule de la grille peut contenir l'un de ces elements :
CELL_FREE = 0          # Cellule libre (le robot peut y aller)
CELL_OBSTACLE = 1      # Obstacle statique (mur, meuble - permanent)
CELL_DYNAMIC_OBS = 2   # Obstacle dynamique (personne, autre robot - se deplace)
CELL_ROBOT = 3         # Position du robot
CELL_GOAL = 4          # Position objectif (destination)

# ============================================================================
# ACTIONS POSSIBLES DU ROBOT
# ============================================================================
# Le robot peut se deplacer dans 8 directions + rester sur place
# Cela correspond aux "coups legaux" dans le vocabulaire MCTS du cours
ACTIONS = {
    0: (-1, 0),   # NORD (haut)
    1: (1, 0),    # SUD (bas)
    2: (0, -1),   # OUEST (gauche)
    3: (0, 1),    # EST (droite)
    4: (-1, -1),  # NORD-OUEST (diagonal haut-gauche)
    5: (-1, 1),   # NORD-EST (diagonal haut-droite)
    6: (1, -1),   # SUD-OUEST (diagonal bas-gauche)
    7: (1, 1),    # SUD-EST (diagonal bas-droite)
    8: (0, 0),    # RESTER SUR PLACE (attendre)
}

ACTION_NAMES = {
    0: "N", 1: "S", 2: "W", 3: "E",
    4: "NW", 5: "NE", 6: "SW", 7: "SE", 8: "WAIT"
}


class DynamicObstacle:
    """
    Represente un obstacle dynamique (personne, autre robot).

    Chaque obstacle dynamique a :
    - Une position (x, y)
    - Une vitesse/direction (dx, dy)
    - Un pattern de mouvement (lineaire, aleatoire, circulaire)

    C'est ce qui rend le probleme STOCHASTIQUE : on ne connait pas exactement
    le futur mouvement des obstacles. C'est pourquoi les methodes Monte Carlo
    sont adaptees (cf. cours, principe fondamental section 1.1).
    """

    def __init__(self, x: int, y: int, dx: int = 0, dy: int = 0,
                 pattern: str = "linear", bounds: Tuple[int, int, int, int] = None):
        """
        Args:
            x, y: Position initiale
            dx, dy: Direction de deplacement (-1, 0, ou 1)
            pattern: Type de mouvement
                - "linear" : va-et-vient en ligne droite
                - "random" : mouvement aleatoire a chaque pas
                - "circular" : mouvement circulaire
            bounds: (x_min, y_min, x_max, y_max) limites de deplacement
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.pattern = pattern
        self.bounds = bounds
        self.initial_x = x
        self.initial_y = y
        self.step_count = 0

    def move(self, grid_size: int, grid: np.ndarray, rng: np.random.Generator = None):
        """
        Deplace l'obstacle selon son pattern.

        Le mouvement des obstacles est ce qui cree l'INCERTITUDE dans notre
        probleme. Pendant les playouts MCTS, on simule ces mouvements pour
        estimer la qualite de chaque action du robot.

        Args:
            grid_size: Taille de la grille
            grid: La grille actuelle (pour eviter les collisions)
            rng: Generateur aleatoire (pour reproductibilite)
        """
        if rng is None:
            rng = np.random.default_rng()

        old_x, old_y = self.x, self.y

        if self.pattern == "linear":
            # Mouvement lineaire : l'obstacle va et vient
            new_x = self.x + self.dx
            new_y = self.y + self.dy

            # Rebondir sur les bords ou les limites
            if self.bounds:
                x_min, y_min, x_max, y_max = self.bounds
            else:
                x_min, y_min, x_max, y_max = 0, 0, grid_size - 1, grid_size - 1

            if new_x < x_min or new_x > x_max:
                self.dx = -self.dx
                new_x = self.x + self.dx
            if new_y < y_min or new_y > y_max:
                self.dy = -self.dy
                new_y = self.y + self.dy

            self.x, self.y = new_x, new_y

        elif self.pattern == "random":
            # Mouvement aleatoire : choisir une direction au hasard
            # C'est le cas le plus difficile pour le robot car imprevisible
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            dx, dy = directions[rng.integers(0, len(directions))]
            new_x = max(0, min(grid_size - 1, self.x + dx))
            new_y = max(0, min(grid_size - 1, self.y + dy))
            self.x, self.y = new_x, new_y

        elif self.pattern == "circular":
            # Mouvement circulaire autour du point initial
            self.step_count += 1
            radius = 2
            angle = self.step_count * np.pi / 4
            self.x = int(self.initial_x + radius * np.cos(angle))
            self.y = int(self.initial_y + radius * np.sin(angle))
            self.x = max(0, min(grid_size - 1, self.x))
            self.y = max(0, min(grid_size - 1, self.y))

        # Verifier que la nouvelle position est valide
        if (0 <= self.x < grid_size and 0 <= self.y < grid_size
                and grid[self.x, self.y] != CELL_OBSTACLE):
            return
        else:
            # Position invalide : revenir en arriere
            self.x, self.y = old_x, old_y

    def copy(self):
        """Copie profonde - CRUCIAL pour MCTS (cf. cours section 25.3 erreur #1)."""
        obs = DynamicObstacle(
            self.x, self.y, self.dx, self.dy,
            self.pattern, self.bounds
        )
        obs.initial_x = self.initial_x
        obs.initial_y = self.initial_y
        obs.step_count = self.step_count
        return obs


class GridEnvironment:
    """
    Environnement de grille 2D pour la planification de trajectoire.

    Cet environnement est l'equivalent de l'etat de jeu dans le cours MCTS.
    Il fournit les methodes requises par le cours (section 25.1) :
    - legal_moves() : actions possibles du robot
    - play(action) : effectuer une action
    - is_terminal() : verifier si l'etat est terminal
    - score() : calculer la recompense
    - copy() : copier l'etat pour les simulations

    L'environnement est un MDP (Markov Decision Process) :
    - Etats : (position_robot, positions_obstacles, temps)
    - Actions : 9 directions possibles
    - Transitions : deterministes pour le robot, stochastiques pour les obstacles
    - Recompenses : +100 objectif, -100 collision, -1 par pas
    """

    def __init__(self, size: int = 20, robot_pos: Tuple[int, int] = None,
                 goal_pos: Tuple[int, int] = None, max_steps: int = 200,
                 sensor_noise: float = 0.0, seed: int = None):
        """
        Args:
            size: Taille de la grille (size x size)
            robot_pos: Position initiale du robot (x, y)
            goal_pos: Position objectif (x, y)
            max_steps: Nombre maximum de pas (evite les boucles infinies - cf. cours erreur #6)
            sensor_noise: Probabilite d'erreur du capteur (0.0 = parfait, 0.3 = 30% de bruit)
            seed: Graine aleatoire pour reproductibilite
        """
        self.size = size
        self.max_steps = max_steps
        self.sensor_noise = sensor_noise
        self.rng = np.random.default_rng(seed)

        # Initialiser la grille vide
        self.grid = np.zeros((size, size), dtype=int)

        # Position robot et objectif
        self.robot_pos = robot_pos if robot_pos else (1, 1)
        self.goal_pos = goal_pos if goal_pos else (size - 2, size - 2)

        # Obstacles dynamiques
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # Compteur de pas
        self.steps = 0
        self.done = False
        self.reached_goal = False

        # Historique des positions du robot (pour la visualisation)
        self.path_history = [self.robot_pos]

    def add_walls_border(self):
        """Ajoute des murs sur les bords de la grille."""
        self.grid[0, :] = CELL_OBSTACLE
        self.grid[-1, :] = CELL_OBSTACLE
        self.grid[:, 0] = CELL_OBSTACLE
        self.grid[:, -1] = CELL_OBSTACLE

    def add_obstacle(self, x: int, y: int):
        """
        Ajoute un obstacle statique a la position (x, y).
        Obstacles statiques = murs, meubles, etc.
        """
        if (x, y) != self.robot_pos and (x, y) != self.goal_pos:
            self.grid[x, y] = CELL_OBSTACLE

    def add_obstacle_rect(self, x1: int, y1: int, x2: int, y2: int):
        """Ajoute un rectangle d'obstacles (pour creer des murs)."""
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                self.add_obstacle(x, y)

    def add_dynamic_obstacle(self, obstacle: DynamicObstacle):
        """Ajoute un obstacle dynamique a l'environnement."""
        self.dynamic_obstacles.append(obstacle)

    def get_grid_state(self) -> np.ndarray:
        """
        Retourne la grille complete avec tous les elements.
        Utilisee pour la visualisation et le hashing (cf. Zobrist, cours section 4).
        """
        grid = self.grid.copy()
        # Placer les obstacles dynamiques
        for obs in self.dynamic_obstacles:
            if 0 <= obs.x < self.size and 0 <= obs.y < self.size:
                if grid[obs.x, obs.y] == CELL_FREE:
                    grid[obs.x, obs.y] = CELL_DYNAMIC_OBS
        # Placer le robot et l'objectif
        rx, ry = self.robot_pos
        gx, gy = self.goal_pos
        grid[gx, gy] = CELL_GOAL
        grid[rx, ry] = CELL_ROBOT
        return grid

    def get_observed_grid(self) -> np.ndarray:
        """
        Retourne la grille telle que le robot la percoit (avec bruit capteur).

        Le bruit capteur simule l'incertitude des capteurs reels :
        - Avec probabilite (1 - sensor_noise), l'observation est correcte
        - Avec probabilite sensor_noise, une cellule libre peut apparaitre
          comme occupee ou vice versa

        Cela est directement lie aux jeux a information imparfaite (cours section 19).
        Le robot ne connait pas l'etat exact de l'environnement.
        """
        grid = self.get_grid_state()
        if self.sensor_noise <= 0:
            return grid

        # Appliquer du bruit
        observed = grid.copy()
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.robot_pos or (i, j) == self.goal_pos:
                    continue  # Le robot connait sa position et l'objectif
                if self.rng.random() < self.sensor_noise:
                    # Bruit : inverser libre <-> obstacle
                    if observed[i, j] == CELL_FREE:
                        observed[i, j] = CELL_DYNAMIC_OBS  # Faux positif
                    elif observed[i, j] in (CELL_DYNAMIC_OBS, CELL_OBSTACLE):
                        observed[i, j] = CELL_FREE  # Faux negatif (dangereux!)
        return observed

    def legal_moves(self) -> List[int]:
        """
        Retourne la liste des actions legales du robot.

        Cette methode est l'equivalent de legal_moves() dans le cours (section 25.1).
        Un coup est legal si :
        1. La case destination est dans la grille
        2. La case destination n'est pas un obstacle statique
        3. La case destination n'est pas occupee par un obstacle dynamique

        Returns:
            Liste d'indices d'actions (0-8)
        """
        legal = []
        rx, ry = self.robot_pos
        for action_id, (dx, dy) in ACTIONS.items():
            new_x, new_y = rx + dx, ry + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size
                    and self.grid[new_x, new_y] != CELL_OBSTACLE):
                # Verifier aussi les obstacles dynamiques
                occupied = False
                for obs in self.dynamic_obstacles:
                    if obs.x == new_x and obs.y == new_y:
                        occupied = True
                        break
                if not occupied:
                    legal.append(action_id)

        # S'il n'y a aucun coup legal, autoriser au moins WAIT
        if not legal:
            legal = [8]
        return legal

    def play(self, action: int):
        """
        Execute une action et fait avancer l'environnement d'un pas de temps.

        Equivalent de play() dans le cours (section 25.1).
        Apres le mouvement du robot, les obstacles dynamiques se deplacent aussi.

        Args:
            action: Indice de l'action (0-8)
        """
        if self.done:
            return

        # 1. Deplacer le robot
        dx, dy = ACTIONS[action]
        new_x = self.robot_pos[0] + dx
        new_y = self.robot_pos[1] + dy

        # Verifier la validite
        if (0 <= new_x < self.size and 0 <= new_y < self.size
                and self.grid[new_x, new_y] != CELL_OBSTACLE):
            self.robot_pos = (new_x, new_y)

        # 2. Deplacer les obstacles dynamiques
        for obs in self.dynamic_obstacles:
            obs.move(self.size, self.grid, self.rng)

        # 3. Verifier les conditions de terminaison
        self.steps += 1
        self.path_history.append(self.robot_pos)

        # Collision avec obstacle dynamique ?
        for obs in self.dynamic_obstacles:
            if self.robot_pos == (obs.x, obs.y):
                self.done = True
                self.reached_goal = False
                return

        # Objectif atteint ?
        if self.robot_pos == self.goal_pos:
            self.done = True
            self.reached_goal = True
            return

        # Nombre maximum de pas atteint ?
        if self.steps >= self.max_steps:
            self.done = True
            self.reached_goal = False

    def is_terminal(self) -> bool:
        """
        Verifie si l'etat est terminal.

        Equivalent de is_terminal() dans le cours (section 25.1).
        Terminal si : objectif atteint, collision, ou max_steps depasse.
        """
        return self.done

    def score(self) -> float:
        """
        Calcule le score/recompense de l'etat actuel.

        Equivalent de score() dans le cours (section 25.1).

        Utilise un score EXPONENTIEL pour creer de grands ecarts
        entre les positions proches et eloignees de l'objectif.
        Cela donne un signal clair a MCTS : chaque pas vers l'objectif
        augmente significativement le score.

        Inspire de la "discounting heuristic" (cours section 20.1) :
        score = gamma^distance, ou gamma < 1.
        """
        if self.reached_goal:
            return 1.0

        # Collision avec obstacle dynamique
        for obs in self.dynamic_obstacles:
            if self.robot_pos == (obs.x, obs.y):
                return -1.0

        rx, ry = self.robot_pos
        gx, gy = self.goal_pos
        current_distance = abs(rx - gx) + abs(ry - gy)

        # Score exponentiel : gamma^distance
        # gamma=0.9 : chaque case de plus divise le score par ~0.9
        # distance 0 -> score 1.0, distance 5 -> 0.59, distance 10 -> 0.35
        # distance 20 -> 0.12, distance 24 -> 0.08
        # Ecart entre distance 23 et 24 : 0.089 vs 0.080 = 10% de difference
        # (vs seulement 3.8% avec score lineaire)
        gamma = 0.9
        score = gamma ** current_distance

        return score

    def heuristic_score(self) -> float:
        """
        Score heuristique rapide (distance a l'objectif normalisee).
        Utilisee dans les playouts pour les evaluations intermediaires.
        """
        rx, ry = self.robot_pos
        gx, gy = self.goal_pos
        distance = abs(rx - gx) + abs(ry - gy)
        max_distance = self.size * 2
        return 1.0 - (distance / max_distance)

    def copy(self):
        """
        Copie profonde de l'environnement.

        CRUCIAL pour MCTS ! (cf. cours section 25.3, erreur #1)
        Chaque simulation/playout doit travailler sur une COPIE de l'etat,
        sinon on modifie l'etat reel.
        """
        env = GridEnvironment(
            size=self.size,
            robot_pos=self.robot_pos,
            goal_pos=self.goal_pos,
            max_steps=self.max_steps,
            sensor_noise=self.sensor_noise
        )
        env.grid = self.grid.copy()
        env.dynamic_obstacles = [obs.copy() for obs in self.dynamic_obstacles]
        env.steps = self.steps
        env.done = self.done
        env.reached_goal = self.reached_goal
        env.rng = np.random.default_rng(self.rng.integers(0, 2**31))
        env.path_history = list(self.path_history)
        return env

    def get_state_hash(self) -> int:
        """
        Calcule un hash de l'etat actuel pour la table de transposition.

        Inspire du Zobrist Hashing (cours section 4.2) :
        On utilise XOR de nombres aleatoires pour chaque element de l'etat.
        Cela permet une mise a jour en O(1) quand l'etat change.
        """
        # Version simplifiee : hash base sur les positions
        state = (
            self.robot_pos,
            tuple((obs.x, obs.y) for obs in self.dynamic_obstacles),
            self.steps
        )
        return hash(state)

    def manhattan_distance_to_goal(self) -> int:
        """Distance Manhattan du robot a l'objectif."""
        rx, ry = self.robot_pos
        gx, gy = self.goal_pos
        return abs(rx - gx) + abs(ry - gy)

    def __repr__(self):
        grid = self.get_grid_state()
        symbols = {
            CELL_FREE: '.', CELL_OBSTACLE: '#',
            CELL_DYNAMIC_OBS: 'D', CELL_ROBOT: 'R', CELL_GOAL: 'G'
        }
        lines = []
        for i in range(self.size):
            line = ' '.join(symbols.get(grid[i, j], '?') for j in range(self.size))
            lines.append(line)
        return '\n'.join(lines)


# ============================================================================
# FONCTIONS DE CREATION DE SCENARIOS
# ============================================================================

def create_simple_scenario(size: int = 20) -> GridEnvironment:
    """
    Scenario 1 : Environnement simple avec quelques obstacles statiques.
    Bon pour tester le MCTS de base.
    """
    env = GridEnvironment(size=size, robot_pos=(1, 1), goal_pos=(size-2, size-2))
    env.add_walls_border()

    # Quelques murs internes (adaptes a la taille)
    s = size
    env.add_obstacle_rect(s//4, 2, s//4, s//2)          # Mur horizontal haut
    env.add_obstacle_rect(s//2, s//3, s//2, 3*s//4)     # Mur horizontal milieu
    env.add_obstacle_rect(2, 3*s//4, s//2-1, 3*s//4)    # Mur vertical

    return env


def create_dynamic_scenario(size: int = 20) -> GridEnvironment:
    """
    Scenario 2 : Environnement avec obstacles statiques ET dynamiques.
    Teste la capacite de MCTS a gerer l'incertitude.
    """
    env = GridEnvironment(size=size, robot_pos=(1, 1), goal_pos=(size-2, size-2))
    env.add_walls_border()

    s = size
    # Obstacles statiques (adaptes a la taille)
    env.add_obstacle_rect(s//4, 2, s//4, s//2)
    env.add_obstacle_rect(s//2, s//3, s//2, 3*s//4)

    # Obstacles dynamiques
    # Obstacle lineaire horizontal
    env.add_dynamic_obstacle(DynamicObstacle(
        x=s//3, y=s//4, dx=0, dy=1, pattern="linear",
        bounds=(s//3, 2, s//3, s-3)
    ))
    # Obstacle lineaire vertical
    env.add_dynamic_obstacle(DynamicObstacle(
        x=3, y=s//2, dx=1, dy=0, pattern="linear",
        bounds=(2, s//2, s//2, s//2)
    ))
    # Obstacle aleatoire (le plus difficile)
    env.add_dynamic_obstacle(DynamicObstacle(
        x=2*s//3, y=2*s//3, pattern="random"
    ))

    return env


def create_complex_scenario(size: int = 30) -> GridEnvironment:
    """
    Scenario 3 : Environnement complexe avec beaucoup d'obstacles.
    Teste les performances et la scalabilite.
    """
    env = GridEnvironment(size=size, robot_pos=(1, 1), goal_pos=(size-2, size-2),
                          max_steps=300)
    env.add_walls_border()

    # Labyrinthe-like structure
    for i in range(4, size-4, 6):
        # Murs horizontaux avec ouvertures
        gap = np.random.randint(2, size-4)
        for j in range(1, size-1):
            if abs(j - gap) > 2:
                env.add_obstacle(i, j)

    # Obstacles dynamiques
    for k in range(5):
        x = np.random.randint(3, size-3)
        y = np.random.randint(3, size-3)
        pattern = np.random.choice(["linear", "random"])
        dx = np.random.choice([-1, 0, 1])
        dy = np.random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            dy = 1
        env.add_dynamic_obstacle(DynamicObstacle(x=x, y=y, dx=dx, dy=dy, pattern=pattern))

    return env


def create_narrow_passage_scenario(size: int = 20) -> GridEnvironment:
    """
    Scenario 4 : Passage etroit avec obstacles dynamiques.
    Teste la capacite de re-planification.
    """
    env = GridEnvironment(size=size, robot_pos=(1, 1), goal_pos=(size-2, size-2))
    env.add_walls_border()

    # Mur avec un seul passage
    for i in range(1, size-1):
        if i != size // 2:
            env.add_obstacle(size // 2, i)

    # Obstacle dynamique qui bloque le passage periodiquement
    env.add_dynamic_obstacle(DynamicObstacle(
        x=size // 2 - 2, y=size // 2, dx=1, dy=0, pattern="linear",
        bounds=(size // 2 - 3, size // 2, size // 2 + 3, size // 2)
    ))

    return env


def create_sensor_noise_scenario(size: int = 20, noise: float = 0.1) -> GridEnvironment:
    """
    Scenario 5 : Environnement avec bruit capteur.
    Teste la robustesse face a l'incertitude d'observation.
    """
    env = GridEnvironment(size=size, robot_pos=(1, 1), goal_pos=(size-2, size-2),
                          sensor_noise=noise)
    env.add_walls_border()

    # Quelques obstacles
    env.add_obstacle_rect(5, 3, 5, 10)
    env.add_obstacle_rect(10, 8, 10, 15)

    # Obstacles dynamiques
    env.add_dynamic_obstacle(DynamicObstacle(
        x=8, y=6, dx=0, dy=1, pattern="linear",
        bounds=(8, 2, 8, 15)
    ))

    return env
