"""
mcts_rave.py - MCTS avec RAVE (Rapid Action Value Estimation) et GRAVE (Generalized RAVE).

RAVE est une amelioration majeure de MCTS qui accelere considerablement
l'apprentissage en utilisant les statistiques AMAF (All Moves As First).

Lien direct avec le cours :
- Section 5.1 : Probleme d'UCT pur (trop lent pour beaucoup d'actions)
- Section 5.2 : AMAF - All Moves As First
- Section 5.3 : RAVE - formule combinant UCT et AMAF
- Section 6.1 : GRAVE - utilise les stats AMAF d'un ancetre plus fiable

Intuition cle :
    "Si un coup m apparait n'importe ou dans une partie gagnante,
     alors m est probablement bon." (cours section 5.2)

Pour notre robot :
    Si une direction (ex: aller au SUD-EST) apparait dans une trajectoire
    reussie, meme si elle n'etait pas le premier mouvement, c'est un signal
    que cette direction est utile pour atteindre l'objectif.

References :
- Gelly & Silver (2007) "Combining Online and Offline Knowledge in UCT" - ICML
- Cazenave (2015) "Generalized Rapid Action Value Estimation" - IJCAI
- Browne et al. (2012) "A Survey of MCTS Methods" - Section IV.D
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .environment import GridEnvironment, ACTIONS, ACTION_NAMES
from .mcts_base import MCTSNode


class MCTS_RAVE:
    """
    MCTS avec RAVE (Rapid Action Value Estimation).

    RAVE combine deux sources d'information (cours section 5.3) :
    1. UCT classique : statistiques exactes du coup comme premier coup
    2. AMAF : statistiques de tous les playouts contenant ce coup

    La formule RAVE (cours equation 5) :
        Valeur_m = (1 - beta_m) * UCT_m + beta_m * AMAF_m

    Le poids beta (cours equation 6) :
        beta_m = p_AMAF_m / (p_AMAF_m + p_m + bias * p_AMAF_m * p_m)

    Comportement de beta :
    - Debut (p_m petit) : beta ~= 1 => AMAF domine (rapide mais imprecis)
    - Fin (p_m grand)   : beta ~= 0 => UCT domine  (lent mais precis)
    => RAVE obtient le meilleur des deux mondes !

    Performances (cours section 5.4) :
    - RAVE bat UCT sur presque tous les jeux :
      Atarigo 94.2%, Go 9x9 73.2%, Domineering 72.6%
    """

    def __init__(self, n_simulations: int = 1000, exploration_constant: float = 1.414,
                 rave_bias: float = 0.0001, max_playout_steps: int = 100,
                 use_heuristic_playout: bool = False, max_depth: int = 15,
                 seed: int = None):
        """
        Args:
            n_simulations: Nombre d'iterations MCTS
            exploration_constant: C dans UCT
            rave_bias: Constante bias dans la formule beta (cours: 0.0001)
            max_playout_steps: Limite de pas dans les playouts
            use_heuristic_playout: Utiliser un playout heuristique
            max_depth: Profondeur max de l'arbre
            seed: Graine aleatoire
        """
        self.n_simulations = n_simulations
        self.C = exploration_constant
        self.rave_bias = rave_bias
        self.max_playout_steps = max_playout_steps
        self.use_heuristic_playout = use_heuristic_playout
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.stats = {}

    def search(self, state: GridEnvironment) -> int:
        """Lance la recherche MCTS-RAVE et retourne la meilleure action."""
        start_time = time.time()
        root = MCTSNode(state.copy())

        for _ in range(self.n_simulations):
            node, depth = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded and depth < self.max_depth:
                node = self._expand(node)

            reward, actions_played = self._simulate_with_actions(node.state)
            self._backpropagate_rave(node, reward, actions_played)

        best_action = self._best_action(root)
        self.stats['search_time'] = time.time() - start_time
        return best_action

    def _select(self, node: MCTSNode):
        """Selection avec score RAVE, limitee en profondeur."""
        depth = 0
        while node.is_fully_expanded and not node.is_terminal and depth < self.max_depth:
            node = self._rave_select_child(node)
            depth += 1
        return node, depth

    def _rave_select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Selectionne l'enfant avec le meilleur score RAVE.

        Formule RAVE (cours section 5.3, equation 5) :
            Valeur_m = (1 - beta_m) * UCT_m + beta_m * AMAF_m

        UCT_m = wi/ni + C * sqrt(ln(N)/ni)  (partie classique)
        AMAF_m = recompense AMAF moyenne du coup m
        beta_m = combine les deux selon le nombre de visites

        Plus un noeud est visite (ni grand), plus on fait confiance a UCT.
        Moins un noeud est visite, plus on utilise AMAF (information rapide).
        """
        best_score = -float('inf')
        best_child = None

        log_parent = math.log(node.visits) if node.visits > 0 else 0

        for action, child in node.children.items():
            if child.visits == 0:
                return child

            # === Score UCT classique ===
            exploitation = child.q_value
            exploration = self.C * math.sqrt(log_parent / child.visits)

            # === Score AMAF ===
            amaf_n = node.amaf_visits.get(action, 0)
            if amaf_n > 0:
                amaf_score = node.amaf_rewards.get(action, 0.0) / amaf_n
            else:
                amaf_score = exploitation  # fallback si pas de stats AMAF

            # === Calcul de beta (cours equation 6) ===
            p_m = child.visits
            p_amaf = amaf_n

            if p_amaf + p_m > 0:
                beta = p_amaf / (p_amaf + p_m + self.rave_bias * p_amaf * p_m)
            else:
                beta = 0.0

            # === Score RAVE final ===
            # IMPORTANT : l'exploration UCT est gardee SEPAREMENT
            # pour ne pas etre ecrasee par AMAF (sinon les noeuds
            # peu visites ne sont jamais re-explores)
            combined_exploitation = (1 - beta) * exploitation + beta * amaf_score
            rave_score = combined_exploitation + exploration

            if rave_score > best_score:
                best_score = rave_score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Phase d'expansion identique a UCT standard."""
        action = node.untried_actions.pop(
            self.rng.integers(0, len(node.untried_actions))
        )
        new_state = node.state.copy()
        new_state.play(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _simulate_with_actions(self, state: GridEnvironment) -> Tuple[float, List[int]]:
        """
        Playout qui retourne le score final + les actions jouees.
        Le score final est plus informatif car il reflete la position
        reellement atteinte apres le playout.
        """
        sim_state = state.copy()
        actions_played = []
        steps = 0

        while not sim_state.is_terminal() and steps < self.max_playout_steps:
            legal = sim_state.legal_moves()

            if self.use_heuristic_playout:
                action = self._heuristic_playout(sim_state, legal)
            else:
                action = legal[self.rng.integers(0, len(legal))]

            actions_played.append(action)
            sim_state.play(action)
            steps += 1

        return sim_state.score(), actions_played

    def _heuristic_playout(self, state: GridEnvironment, legal: List[int]) -> int:
        """Politique de playout heuristique forte (temperature=3)."""
        rx, ry = state.robot_pos
        gx, gy = state.goal_pos
        temperature = 1.0

        scores = []
        for action in legal:
            dx, dy = ACTIONS[action]
            new_x, new_y = rx + dx, ry + dy
            dist = abs(new_x - gx) + abs(new_y - gy)
            if action == 8 and len(legal) > 1:
                dist += 2
            scores.append(-dist * temperature)

        scores = np.array(scores, dtype=float)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        return legal[self.rng.choice(len(legal), p=probs)]

    def _backpropagate_rave(self, node: MCTSNode, reward: float,
                             actions_played: List[int]):
        """
        Retropropagation avec mise a jour AMAF (pour RAVE).

        En plus de mettre a jour les statistiques UCT classiques (N, W),
        on met aussi a jour les statistiques AMAF :
        - Pour chaque noeud sur le chemin racine -> feuille,
        - Pour chaque action jouee pendant le playout,
        - On met a jour amaf_visits et amaf_rewards

        C'est ce qui permet a RAVE d'apprendre beaucoup plus vite :
        un seul playout donne de l'information sur TOUTES les actions
        qu'il contient (cours section 5.2).
        """
        # Convertir en ensemble pour verification rapide
        actions_set = set(actions_played)

        while node is not None:
            # Mise a jour UCT classique
            node.visits += 1
            node.total_reward += reward

            # Mise a jour AMAF
            # Pour chaque action jouee dans le playout, mettre a jour
            # les statistiques AMAF de ce noeud
            for action in actions_set:
                node.amaf_visits[action] = node.amaf_visits.get(action, 0) + 1
                node.amaf_rewards[action] = node.amaf_rewards.get(action, 0.0) + reward

            node = node.parent

    def _best_action(self, root: MCTSNode) -> int:
        """Choisir l'action avec la meilleure valeur Q (pour navigation)."""
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


class MCTS_GRAVE:
    """
    MCTS avec GRAVE (Generalized RAVE).

    GRAVE ameliore RAVE en resolant un probleme important :
    quand un noeud a peu de playouts, ses statistiques AMAF ne sont pas fiables.

    Solution GRAVE (cours section 6.1) :
        Au lieu d'utiliser les statistiques AMAF du noeud courant,
        on utilise celles du premier ANCETRE ayant plus de n_ref playouts.

    Justification :
        Un ancetre avec 100+ playouts a des statistiques AMAF plus fiables
        qu'un noeud avec 5 playouts.

    Performances GRAVE vs RAVE (cours section 6.2) :
    - Atarigo 88.4%, Domineering 62.4%, Knightthrough 67.2%

    Note : GRAVE avec n_ref = 0 est equivalent a RAVE.

    Reference :
    - Cazenave (2015) "Generalized Rapid Action Value Estimation" - IJCAI
    """

    def __init__(self, n_simulations: int = 1000, exploration_constant: float = 1.414,
                 rave_bias: float = 0.0001, grave_threshold: int = 50,
                 max_playout_steps: int = 100, use_heuristic_playout: bool = False,
                 max_depth: int = 15, seed: int = None):
        """
        Args:
            n_simulations: Nombre d'iterations MCTS
            exploration_constant: C dans UCT
            rave_bias: Constante bias pour beta
            grave_threshold: Seuil n_ref pour GRAVE (cours: 50)
            max_playout_steps: Limite de pas
            use_heuristic_playout: Playout heuristique
            max_depth: Profondeur max de l'arbre
            seed: Graine aleatoire
        """
        self.n_simulations = n_simulations
        self.C = exploration_constant
        self.rave_bias = rave_bias
        self.grave_threshold = grave_threshold
        self.max_playout_steps = max_playout_steps
        self.use_heuristic_playout = use_heuristic_playout
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.stats = {}

    def search(self, state: GridEnvironment) -> int:
        """Lance la recherche MCTS-GRAVE."""
        start_time = time.time()
        root = MCTSNode(state.copy())

        for _ in range(self.n_simulations):
            node, depth = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded and depth < self.max_depth:
                node = self._expand(node)

            reward, actions_played = self._simulate_with_actions(node.state)
            self._backpropagate_rave(node, reward, actions_played)

        best_action = self._best_action(root)
        self.stats['search_time'] = time.time() - start_time
        return best_action

    def _find_grave_ancestor(self, node: MCTSNode) -> MCTSNode:
        """
        Trouve le premier ancetre avec assez de playouts pour GRAVE.

        C'est l'innovation cle de GRAVE (cours section 6.1) :
        Au lieu d'utiliser les stats AMAF du noeud courant (qui peut etre
        imprecis si peu visite), on remonte dans l'arbre jusqu'a trouver
        un ancetre avec au moins grave_threshold playouts.

        Exemple :
        - Noeud courant : 5 visites, AMAF imprecis
        - Parent : 20 visites, AMAF encore imprecis
        - Grand-parent : 100 visites => on utilise ses stats AMAF !
        """
        ancestor = node
        while ancestor is not None:
            if ancestor.visits >= self.grave_threshold:
                return ancestor
            ancestor = ancestor.parent
        return node  # Fallback : utiliser le noeud lui-meme

    def _select(self, node: MCTSNode):
        """Selection avec score GRAVE, limitee en profondeur."""
        depth = 0
        while node.is_fully_expanded and not node.is_terminal and depth < self.max_depth:
            node = self._grave_select_child(node)
            depth += 1
        return node, depth

    def _grave_select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Selectionne l'enfant avec le meilleur score GRAVE.

        La difference avec RAVE est dans le calcul d'AMAF :
        - RAVE : utilise node.amaf_visits[action] (stats du noeud courant)
        - GRAVE : utilise ancestor.amaf_visits[action] (stats de l'ancetre fiable)
        """
        best_score = -float('inf')
        best_child = None

        log_parent = math.log(node.visits) if node.visits > 0 else 0

        # Trouver l'ancetre de reference pour GRAVE
        ref_ancestor = self._find_grave_ancestor(node)

        for action, child in node.children.items():
            if child.visits == 0:
                return child

            # Score UCT classique
            exploitation = child.q_value
            exploration = self.C * math.sqrt(log_parent / child.visits)

            # Score AMAF depuis l'ancetre de reference (GRAVE)
            amaf_n = ref_ancestor.amaf_visits.get(action, 0)
            if amaf_n > 0:
                amaf_score = ref_ancestor.amaf_rewards.get(action, 0.0) / amaf_n
            else:
                amaf_score = exploitation  # fallback si pas de stats AMAF

            # Beta avec les stats de l'ancetre
            p_m = child.visits
            p_amaf = amaf_n
            if p_amaf + p_m > 0:
                beta = p_amaf / (p_amaf + p_m + self.rave_bias * p_amaf * p_m)
            else:
                beta = 0.0

            # Score GRAVE final
            # IMPORTANT : exploration UCT gardee separement (pas ecrasee par AMAF)
            combined_exploitation = (1 - beta) * exploitation + beta * amaf_score
            grave_score = combined_exploitation + exploration

            if grave_score > best_score:
                best_score = grave_score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion standard."""
        action = node.untried_actions.pop(
            self.rng.integers(0, len(node.untried_actions))
        )
        new_state = node.state.copy()
        new_state.play(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _simulate_with_actions(self, state: GridEnvironment) -> Tuple[float, List[int]]:
        """Playout retournant score final + actions jouees (pour AMAF)."""
        sim_state = state.copy()
        actions_played = []
        steps = 0

        while not sim_state.is_terminal() and steps < self.max_playout_steps:
            legal = sim_state.legal_moves()
            if self.use_heuristic_playout:
                action = self._heuristic_playout(sim_state, legal)
            else:
                action = legal[self.rng.integers(0, len(legal))]
            actions_played.append(action)
            sim_state.play(action)
            steps += 1

        return sim_state.score(), actions_played

    def _heuristic_playout(self, state: GridEnvironment, legal: List[int]) -> int:
        """Politique de playout heuristique forte (temperature=3)."""
        rx, ry = state.robot_pos
        gx, gy = state.goal_pos
        temperature = 1.0
        scores = []
        for action in legal:
            dx, dy = ACTIONS[action]
            new_x, new_y = rx + dx, ry + dy
            dist = abs(new_x - gx) + abs(new_y - gy)
            if action == 8 and len(legal) > 1:
                dist += 2
            scores.append(-dist * temperature)
        scores = np.array(scores, dtype=float)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        return legal[self.rng.choice(len(legal), p=probs)]

    def _backpropagate_rave(self, node: MCTSNode, reward: float,
                             actions_played: List[int]):
        """Retropropagation avec AMAF (identique a RAVE)."""
        actions_set = set(actions_played)
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            for action in actions_set:
                node.amaf_visits[action] = node.amaf_visits.get(action, 0) + 1
                node.amaf_rewards[action] = node.amaf_rewards.get(action, 0.0) + reward
            node = node.parent

    def _best_action(self, root: MCTSNode) -> int:
        """Choisir l'action avec la meilleure valeur Q (pour navigation)."""
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
