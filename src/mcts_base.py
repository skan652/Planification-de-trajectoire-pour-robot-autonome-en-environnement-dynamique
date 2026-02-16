"""
mcts_base.py - Implementation de MCTS de base avec UCT pour la planification de trajectoire.

Ce module implemente l'algorithme MCTS classique tel que decrit dans le cours :
- Les 4 phases : Selection, Expansion, Simulation, Retropropagation
- La formule UCT pour l'equilibre exploration/exploitation
- La table de transposition pour stocker les statistiques

Lien direct avec le cours :
- Section 3.1 : Les 4 phases de MCTS
- Section 3.2 : Formule UCT
- Section 3.3 : Algorithme UCT recursif
- Section 4 : Table de transposition et Zobrist Hashing
- Section 16 : Progressive Widening (pour les actions continues)

References :
- Kocsis & Szepesvari (2006) "Bandit Based Monte-Carlo Planning" - ECML 2006
- Browne et al. (2012) "A Survey of MCTS Methods" - IEEE TCIAIG
- Dam et al. (2022) "Monte-Carlo Robot Path Planning" - IEEE RA-L
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .environment import GridEnvironment, ACTIONS, ACTION_NAMES


class MCTSNode:
    """
    Noeud de l'arbre MCTS.

    Chaque noeud represente un etat de l'environnement et contient :
    - Les statistiques de visites (N) et de recompenses (W)
    - Les enfants (un par action possible)
    - Le parent (pour la retropropagation)

    Cela correspond a la structure decrite dans le cours (section 4.4) :
    - Nombre total de playouts : N
    - Pour chaque coup possible m : nombre de playouts nm, nombre de victoires wm

    Reference : Browne et al. (2012), Section III.A
    """

    def __init__(self, state: GridEnvironment, parent=None, action: int = None):
        """
        Args:
            state: L'etat de l'environnement a ce noeud
            parent: Le noeud parent (None pour la racine)
            action: L'action qui a mene a ce noeud depuis le parent
        """
        self.state = state
        self.parent = parent
        self.action = action

        # Statistiques (coeur de MCTS)
        self.visits = 0          # N : nombre de visites (playouts passant par ce noeud)
        self.total_reward = 0.0  # W : somme des recompenses

        # Enfants
        self.children: Dict[int, 'MCTSNode'] = {}  # action -> noeud enfant
        self.untried_actions: List[int] = state.legal_moves() if not state.is_terminal() else []

        # Pour RAVE (sera utilise dans mcts_rave.py)
        self.amaf_visits: Dict[int, int] = defaultdict(int)
        self.amaf_rewards: Dict[int, float] = defaultdict(float)

    @property
    def is_fully_expanded(self) -> bool:
        """Verifie si tous les enfants ont ete explores au moins une fois."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        """Verifie si ce noeud est un etat terminal."""
        return self.state.is_terminal()

    @property
    def q_value(self) -> float:
        """
        Valeur Q = recompense moyenne.
        Equivalent de wi/ni dans la formule UCT du cours (section 3.2).
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


class FlatMonteCarlo:
    """
    Flat Monte Carlo - L'algorithme le plus simple (baseline).

    Comme decrit dans le cours (section 1.2) :
    Pour chaque coup possible :
        1. Jouer ce coup
        2. Faire N playouts aleatoires
        3. Compter combien menent a la victoire
    Choisir le coup avec le plus de victoires.

    Avantages : Simple, facile a implementer
    Inconvenients : Lent, pas de memoire entre les coups

    Nous l'utilisons comme BASELINE pour comparer avec les autres algorithmes.
    """

    def __init__(self, n_simulations: int = 1000, max_playout_steps: int = 100,
                 seed: int = None):
        self.n_simulations = n_simulations
        self.max_playout_steps = max_playout_steps
        self.rng = np.random.default_rng(seed)

    def search(self, state: GridEnvironment) -> int:
        """
        Recherche Flat Monte Carlo.

        Pour chaque action legale, on fait n_simulations/nb_actions playouts
        et on choisit l'action avec la meilleure recompense moyenne.
        """
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]

        sims_per_action = max(1, self.n_simulations // len(legal))
        best_action = legal[0]
        best_reward = -float('inf')

        for action in legal:
            total_reward = 0.0
            for _ in range(sims_per_action):
                # Copier l'etat (CRUCIAL - cf. cours erreur #1)
                sim_state = state.copy()
                sim_state.play(action)

                # Playout aleatoire
                reward = self._random_playout(sim_state)
                total_reward += reward

            avg_reward = total_reward / sims_per_action
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        return best_action

    def _random_playout(self, state: GridEnvironment) -> float:
        """
        Playout aleatoire retournant le score final.
        """
        steps = 0
        while not state.is_terminal() and steps < self.max_playout_steps:
            legal = state.legal_moves()
            action = legal[self.rng.integers(0, len(legal))]
            state.play(action)
            steps += 1
        return state.score()


class MCTS_UCT:
    """
    Monte Carlo Tree Search avec UCT (Upper Confidence Bound for Trees).

    C'est l'algorithme principal du cours (sections 3.1-3.3).

    Les 4 phases sont repetees N fois :
    1. SELECTION : Descendre dans l'arbre en choisissant les noeuds avec le
       meilleur score UCT (equilibre exploration/exploitation)
    2. EXPANSION : Ajouter un nouveau noeud enfant pour une action non exploree
    3. SIMULATION : Faire un playout aleatoire depuis le nouveau noeud
    4. RETROPROPAGATION : Remonter le resultat et mettre a jour les statistiques

    Formule UCT (cours section 3.2) :
        UCT = wi/ni + C * sqrt(ln(N) / ni)
    ou :
    - wi = somme des recompenses apres le coup i
    - ni = nombre de visites du coup i
    - N  = nombre de visites du noeud parent
    - C  = constante d'exploration (typiquement sqrt(2))

    References :
    - Kocsis & Szepesvari (2006) - Article fondateur de UCT
    - Dam et al. (2022) - Preuves de convergence pour path planning robot
    """

    def __init__(self, n_simulations: int = 1000, exploration_constant: float = 1.414,
                 max_playout_steps: int = 100, use_heuristic_playout: bool = False,
                 max_depth: int = 15, seed: int = None):
        """
        Args:
            n_simulations: Nombre de simulations MCTS (typiquement 1000-10000)
            exploration_constant: C dans la formule UCT (sqrt(2) par defaut)
            max_playout_steps: Limite de pas dans les playouts
            use_heuristic_playout: Si True, utilise une heuristique dans les playouts
            max_depth: Profondeur maximale de l'arbre (concentre le budget
                sur les decisions les plus proches, crucial pour la navigation)
            seed: Graine aleatoire
        """
        self.n_simulations = n_simulations
        self.C = exploration_constant
        self.max_playout_steps = max_playout_steps
        self.use_heuristic_playout = use_heuristic_playout
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

        # Statistiques pour l'analyse
        self.stats = {
            'total_simulations': 0,
            'avg_playout_length': 0,
            'search_time': 0,
        }

    def search(self, state: GridEnvironment) -> int:
        """
        Lance la recherche MCTS et retourne la meilleure action.

        C'est la boucle principale qui repete les 4 phases N fois.
        A la fin, on choisit l'action avec le plus de visites (plus robuste
        que de choisir celle avec la meilleure valeur Q).

        Args:
            state: L'etat actuel de l'environnement

        Returns:
            L'indice de la meilleure action (0-8)
        """
        start_time = time.time()

        # Creer le noeud racine
        root = MCTSNode(state.copy())

        # Boucle principale : repeter les 4 phases
        for sim in range(self.n_simulations):
            # Phase 1 : SELECTION (avec limite de profondeur)
            node = self._select(root)

            # Phase 2 : EXPANSION (seulement si pas trop profond)
            depth = self._node_depth(node)
            if not node.is_terminal and not node.is_fully_expanded and depth < self.max_depth:
                node = self._expand(node)

            # Phase 3 : SIMULATION (playout)
            reward = self._simulate(node.state)

            # Phase 4 : RETROPROPAGATION
            self._backpropagate(node, reward)

        # Choisir la meilleure action (celle avec le plus de visites)
        best_action = self._best_action(root)

        # Mettre a jour les statistiques
        self.stats['total_simulations'] = self.n_simulations
        self.stats['search_time'] = time.time() - start_time

        return best_action

    def _node_depth(self, node: MCTSNode) -> int:
        """Calcule la profondeur d'un noeud dans l'arbre."""
        depth = 0
        n = node
        while n.parent is not None:
            depth += 1
            n = n.parent
        return depth

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Phase 1 : SELECTION

        Descendre dans l'arbre en suivant la politique UCT.
        S'arrete aussi quand la profondeur max est atteinte,
        ce qui concentre le budget sur les actions proches.
        """
        depth = 0
        while node.is_fully_expanded and not node.is_terminal and depth < self.max_depth:
            node = self._uct_select_child(node)
            depth += 1
        return node

    def _uct_select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Selectionne l'enfant avec le meilleur score UCT.

        Formule UCT (cours section 3.2) :
            UCT_i = Q_i + C * sqrt(ln(N_parent) / N_i)

        ou :
        - Q_i = recompense moyenne du noeud enfant i = wi/ni
        - N_parent = nombre de visites du noeud parent
        - N_i = nombre de visites du noeud enfant i
        - C = constante d'exploration

        Le terme d'exploration sqrt(ln(N)/ni) vient directement de UCB1
        (cours section 2.2) : il est grand quand ni est petit (peu visite)
        et diminue quand ni augmente.
        """
        best_score = -float('inf')
        best_child = None

        log_parent_visits = math.log(node.visits) if node.visits > 0 else 0

        for child in node.children.values():
            if child.visits == 0:
                # Noeud jamais visite => score infini (on l'explore en priorite)
                return child

            # Formule UCT
            exploitation = child.q_value  # Q_i = wi/ni
            exploration = self.C * math.sqrt(log_parent_visits / child.visits)
            uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Phase 2 : EXPANSION

        Choisir une action non exploree et creer un nouveau noeud enfant.
        Cela agrandit l'arbre de recherche d'un noeud.

        Le choix de l'action a expandre peut etre :
        - Aleatoire (standard)
        - Guide par une heuristique (plus efficace)
        """
        # Choisir une action non exploree
        action = node.untried_actions.pop(
            self.rng.integers(0, len(node.untried_actions))
        )

        # Creer le nouvel etat
        new_state = node.state.copy()
        new_state.play(action)

        # Creer le noeud enfant
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child

        return child

    def _simulate(self, state: GridEnvironment) -> float:
        """
        Phase 3 : SIMULATION (playout)

        Retourne le score final du playout. Cela donne un signal
        informatif a MCTS : les actions qui menent vers des regions
        proches de l'objectif auront un meilleur score final.

        Avec le score exponentiel (gamma^distance), chaque case
        de difference se traduit par un ecart significatif.
        """
        sim_state = state.copy()
        steps = 0

        while not sim_state.is_terminal() and steps < self.max_playout_steps:
            legal = sim_state.legal_moves()

            if self.use_heuristic_playout:
                action = self._heuristic_playout_policy(sim_state, legal)
            else:
                action = legal[self.rng.integers(0, len(legal))]

            sim_state.play(action)
            steps += 1

        return sim_state.score()

    def _heuristic_playout_policy(self, state: GridEnvironment,
                                   legal_actions: List[int]) -> int:
        """
        Politique de playout heuristique FORTE.

        Inspiree de PPA (Playout Policy Adaptation, cours section 13)
        et de l'echantillonnage de Gibbs (cours section 10.2).

        On utilise un TEMPERATURE elevee (facteur 3.0) dans le softmax
        pour favoriser fortement les actions qui rapprochent de l'objectif.
        Cela donne des playouts qui vont globalement dans la bonne direction
        tout en gardant un peu d'exploration pour contourner les obstacles.

        P(action) = exp(-3 * distance) / sum(exp(-3 * distances))

        Avec temperature=3 :
        - Action qui rapproche de 1 case : probabilite ~20x plus grande
        - Action qui eloigne de 1 case : probabilite ~20x plus petite
        """
        rx, ry = state.robot_pos
        gx, gy = state.goal_pos
        temperature = 1.0  # Biais modere vers l'objectif, conserve de l'exploration

        scores = []
        for action in legal_actions:
            dx, dy = ACTIONS[action]
            new_x, new_y = rx + dx, ry + dy
            # Distance Manhattan apres l'action
            dist = abs(new_x - gx) + abs(new_y - gy)
            # Penaliser aussi l'action WAIT quand on n'est pas bloque
            if action == 8 and len(legal_actions) > 1:
                dist += 2  # Penalite pour rester immobile
            scores.append(-dist * temperature)

        # Softmax (Gibbs sampling, cf. cours section 10.2)
        scores = np.array(scores, dtype=float)
        scores -= scores.max()  # Stabilite numerique
        probs = np.exp(scores)
        probs /= probs.sum()

        return legal_actions[self.rng.choice(len(legal_actions), p=probs)]

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Phase 4 : RETROPROPAGATION

        Remonter le resultat du playout depuis le noeud feuille jusqu'a la racine.
        A chaque noeud sur le chemin :
        - Incrementer le compteur de visites (N)
        - Ajouter la recompense (W)

        C'est la phase qui met a jour les statistiques utilisees par UCT.
        Sans cette phase, l'arbre n'apprendrait rien des simulations.

        Note : Pour un probleme a un joueur (notre cas), la recompense est
        la meme a chaque niveau. Pour un jeu a 2 joueurs, il faudrait inverser
        le signe a chaque niveau (cf. cours section 25.3, erreur #2).
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_action(self, root: MCTSNode) -> int:
        """
        Choisir la meilleure action apres la recherche.

        Pour la planification de trajectoire (probleme a 1 joueur),
        on choisit l'action avec la meilleure valeur Q (recompense moyenne).
        Cela donne de meilleurs resultats que le choix par visites car
        les scores sont des estimations de distance, pas des probabilites
        de victoire binaires.

        Note : pour les jeux a 2 joueurs, le choix par visites
        est plus robuste (cf. Browne et al. 2012).
        """
        best_q = -float('inf')
        best_action = None

        for action, child in root.children.items():
            if child.visits > 0 and child.q_value > best_q:
                best_q = child.q_value
                best_action = action

        if best_action is None:
            legal = root.state.legal_moves()
            best_action = legal[self.rng.integers(0, len(legal))]

        return best_action

    def get_action_statistics(self, state: GridEnvironment) -> Dict:
        """
        Retourne les statistiques detaillees de la recherche.
        Utile pour le debugging et l'analyse.
        """
        root = MCTSNode(state.copy())

        for _ in range(self.n_simulations):
            node = self._select(root)
            if not node.is_terminal and not node.is_fully_expanded:
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        stats = {}
        for action, child in root.children.items():
            stats[ACTION_NAMES[action]] = {
                'visits': child.visits,
                'q_value': round(child.q_value, 3),
                'total_reward': round(child.total_reward, 3),
            }

        return stats


class MCTS_UCT_TranspositionTable:
    """
    MCTS avec Table de Transposition.

    Au lieu de stocker un arbre, on utilise une table de hachage.
    Cela permet de reconnaitre les etats deja visites meme s'ils sont
    atteints par des chemins differents.

    Lien avec le cours :
    - Section 4 : Table de Transposition et Zobrist Hashing
    - Section 4.4 : Structure de la table (hash -> statistiques)

    Reference : Browne et al. (2012), Section IV.C
    """

    def __init__(self, n_simulations: int = 1000, exploration_constant: float = 1.414,
                 max_playout_steps: int = 100, seed: int = None):
        self.n_simulations = n_simulations
        self.C = exploration_constant
        self.max_playout_steps = max_playout_steps
        self.rng = np.random.default_rng(seed)

        # Table de transposition (cours section 4.4)
        # Cle : hash de l'etat
        # Valeur : dictionnaire avec N, et pour chaque action (n_a, w_a)
        self.table: Dict[int, Dict] = {}

    def _get_or_create_entry(self, state: GridEnvironment) -> Dict:
        """
        Recupere ou cree une entree dans la table de transposition.

        Structure (cf. cours section 4.4) :
        {
            'N': nombre total de visites,
            'actions': {
                action_id: {'n': visites, 'w': recompenses},
                ...
            }
        }
        """
        h = state.get_state_hash()
        if h not in self.table:
            self.table[h] = {
                'N': 0,
                'actions': {}
            }
        return self.table[h]

    def search(self, state: GridEnvironment) -> int:
        """Recherche MCTS avec table de transposition."""
        self.table.clear()

        for _ in range(self.n_simulations):
            sim_state = state.copy()
            reward = self._uct_search(sim_state)

        # Choisir l'action la plus visitee
        entry = self._get_or_create_entry(state)
        best_action = max(
            entry['actions'].keys(),
            key=lambda a: entry['actions'][a]['n'],
            default=state.legal_moves()[0]
        )
        return best_action

    def _uct_search(self, state: GridEnvironment) -> float:
        """
        Algorithme UCT recursif (cours section 3.3, Algorithm 1).

        Pseudo-code du cours :
        if etat est terminal: return score(etat)
        if etat pas dans la table:
            Ajouter etat a la table
            return playout aleatoire(etat)
        m* = argmax_m UCT(m)
        Jouer m*
        r = UCT(nouvel etat)
        Mettre a jour statistiques de m*
        return r
        """
        # Etat terminal
        if state.is_terminal():
            return state.score()

        h = state.get_state_hash()

        # Etat pas dans la table => expansion + playout
        if h not in self.table:
            self.table[h] = {'N': 0, 'actions': {}}
            # Playout aleatoire
            sim = state.copy()
            steps = 0
            while not sim.is_terminal() and steps < self.max_playout_steps:
                legal = sim.legal_moves()
                action = legal[self.rng.integers(0, len(legal))]
                sim.play(action)
                steps += 1
            reward = sim.score()
            self.table[h]['N'] += 1
            return reward

        entry = self.table[h]
        legal = state.legal_moves()

        # Initialiser les actions manquantes
        for a in legal:
            if a not in entry['actions']:
                entry['actions'][a] = {'n': 0, 'w': 0.0}

        # Selectionner la meilleure action selon UCT
        best_action = self._select_uct_action(entry, legal)

        # Jouer l'action
        state.play(best_action)

        # Recursion
        reward = self._uct_search(state)

        # Mise a jour des statistiques
        entry['N'] += 1
        entry['actions'][best_action]['n'] += 1
        entry['actions'][best_action]['w'] += reward

        return reward

    def _select_uct_action(self, entry: Dict, legal: List[int]) -> int:
        """
        Selectionne l'action avec le meilleur score UCT.

        Formule UCT (cours section 3.2) :
            UCT_a = w_a/n_a + C * sqrt(ln(N) / n_a)
        """
        N = entry['N']
        best_score = -float('inf')
        best_action = legal[0]

        log_N = math.log(N) if N > 0 else 0

        for a in legal:
            stats = entry['actions'].get(a, {'n': 0, 'w': 0.0})
            if stats['n'] == 0:
                return a  # Priorite aux actions non visitees

            exploitation = stats['w'] / stats['n']
            exploration = self.C * math.sqrt(log_N / stats['n'])
            uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_action = a

        return best_action
