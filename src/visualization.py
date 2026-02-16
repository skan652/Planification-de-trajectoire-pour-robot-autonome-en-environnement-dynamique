"""
visualization.py - Visualisation de l'environnement et des resultats.

Ce module fournit des outils de visualisation pour :
1. Afficher la grille avec le robot, les obstacles et l'objectif
2. Animer la trajectoire du robot en temps reel
3. Tracer les statistiques de performance (comparaison d'algorithmes)
4. Visualiser l'arbre MCTS (pour le debugging)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional
from .environment import (GridEnvironment, CELL_FREE, CELL_OBSTACLE,
                           CELL_DYNAMIC_OBS, CELL_ROBOT, CELL_GOAL,
                           ACTION_NAMES)


# Couleurs pour la visualisation
COLORS = {
    CELL_FREE: '#FFFFFF',        # Blanc - cellule libre
    CELL_OBSTACLE: '#2C3E50',    # Bleu fonce - obstacle statique
    CELL_DYNAMIC_OBS: '#E74C3C', # Rouge - obstacle dynamique
    CELL_ROBOT: '#3498DB',       # Bleu - robot
    CELL_GOAL: '#2ECC71',        # Vert - objectif
}

CUSTOM_CMAP = ListedColormap([
    COLORS[CELL_FREE],
    COLORS[CELL_OBSTACLE],
    COLORS[CELL_DYNAMIC_OBS],
    COLORS[CELL_ROBOT],
    COLORS[CELL_GOAL],
])


def plot_environment(env: GridEnvironment, title: str = "Environnement",
                      path: List[Tuple[int, int]] = None,
                      save_path: str = None, show: bool = True):
    """
    Affiche l'environnement avec la grille, les obstacles, le robot et l'objectif.

    Args:
        env: L'environnement a afficher
        title: Titre du graphique
        path: Trajectoire a afficher (liste de (x, y))
        save_path: Chemin pour sauvegarder l'image
        show: Afficher la figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    grid = env.get_grid_state()
    ax.imshow(grid, cmap=CUSTOM_CMAP, vmin=0, vmax=4, origin='upper')

    # Dessiner la trajectoire
    if path and len(path) > 1:
        path_x = [p[1] for p in path]  # colonnes
        path_y = [p[0] for p in path]  # lignes
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Trajectoire')
        ax.plot(path_x[0], path_y[0], 'bs', markersize=12, label='Depart')
        ax.plot(path_x[-1], path_y[-1], 'g*', markersize=15, label='Arrivee')

    # Legende
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[CELL_FREE], edgecolor='gray', label='Libre'),
        mpatches.Patch(facecolor=COLORS[CELL_OBSTACLE], label='Obstacle statique'),
        mpatches.Patch(facecolor=COLORS[CELL_DYNAMIC_OBS], label='Obstacle dynamique'),
        mpatches.Patch(facecolor=COLORS[CELL_ROBOT], label='Robot'),
        mpatches.Patch(facecolor=COLORS[CELL_GOAL], label='Objectif'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Grille
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def animate_trajectory(env: GridEnvironment, path: List[Tuple[int, int]],
                        dynamic_positions: List[List[Tuple[int, int]]] = None,
                        save_path: str = None, interval: int = 200):
    """
    Anime la trajectoire du robot pas a pas.

    Args:
        env: L'environnement
        path: Trajectoire du robot
        dynamic_positions: Positions des obstacles dynamiques a chaque pas
        save_path: Chemin pour sauvegarder l'animation (GIF)
        interval: Intervalle entre les frames (ms)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    def update(frame):
        ax.clear()

        # Grille de base (obstacles statiques)
        grid = env.grid.copy()

        # Obstacles dynamiques a ce pas de temps
        if dynamic_positions and frame < len(dynamic_positions):
            for pos in dynamic_positions[frame]:
                if 0 <= pos[0] < env.size and 0 <= pos[1] < env.size:
                    grid[pos[0], pos[1]] = CELL_DYNAMIC_OBS

        # Objectif
        gx, gy = env.goal_pos
        grid[gx, gy] = CELL_GOAL

        # Robot a la position actuelle
        if frame < len(path):
            rx, ry = path[frame]
            grid[rx, ry] = CELL_ROBOT

        ax.imshow(grid, cmap=CUSTOM_CMAP, vmin=0, vmax=4, origin='upper')

        # Tracer le chemin parcouru
        if frame > 0:
            past_x = [p[1] for p in path[:frame+1]]
            past_y = [p[0] for p in path[:frame+1]]
            ax.plot(past_x, past_y, 'b-', linewidth=2, alpha=0.5)

        ax.set_title(f"Step {frame}/{len(path)-1}", fontsize=14)
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)

    ani = animation.FuncAnimation(fig, update, frames=len(path),
                                   interval=interval, repeat=True)

    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
    plt.show()


def plot_comparison(results: Dict[str, List[Dict]], metric: str = 'success_rate',
                     title: str = None, save_path: str = None, show: bool = True):
    """
    Compare les performances de differents algorithmes.

    Args:
        results: {nom_algo: [liste de resultats d'executions]}
        metric: Metrique a comparer :
            - 'success_rate' : taux de succes
            - 'avg_steps' : nombre moyen de pas
            - 'avg_search_time' : temps moyen de recherche
        title: Titre du graphique
        save_path: Chemin pour sauvegarder
        show: Afficher
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    algo_names = list(results.keys())
    values = []
    errors = []

    for algo in algo_names:
        runs = results[algo]
        if metric == 'success_rate':
            vals = [1.0 if r['success'] else 0.0 for r in runs]
        elif metric == 'avg_steps':
            vals = [r['steps'] for r in runs if r['success']]
            if not vals:
                vals = [0]
        elif metric == 'avg_search_time':
            vals = [r.get('avg_search_time', 0) for r in runs]
        else:
            vals = [r.get(metric, 0) for r in runs]

        values.append(np.mean(vals))
        errors.append(np.std(vals) / max(1, np.sqrt(len(vals))))

    bars = ax.bar(algo_names, values, yerr=errors, capsize=5,
                   color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(algo_names)])

    metric_labels = {
        'success_rate': 'Taux de succes',
        'avg_steps': 'Nombre moyen de pas (succes uniquement)',
        'avg_search_time': 'Temps moyen de recherche (s)',
    }

    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(title or f'Comparaison : {metric}', fontsize=14)
    ax.set_ylim(bottom=0)

    # Valeurs sur les barres
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_comparison(results: Dict[str, List[Dict]],
                           save_path: str = None, show: bool = True):
    """
    Affiche une comparaison multi-metriques (3 graphiques).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    algo_names = list(results.keys())
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(algo_names)]

    # 1. Taux de succes
    success_rates = []
    for algo in algo_names:
        rates = [1.0 if r['success'] else 0.0 for r in results[algo]]
        success_rates.append(np.mean(rates))

    axes[0].bar(algo_names, success_rates, color=colors)
    axes[0].set_title('Taux de succes', fontsize=12)
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)

    # 2. Nombre moyen de pas (succes uniquement)
    avg_steps = []
    for algo in algo_names:
        steps = [r['steps'] for r in results[algo] if r['success']]
        avg_steps.append(np.mean(steps) if steps else 0)

    axes[1].bar(algo_names, avg_steps, color=colors)
    axes[1].set_title('Nombre moyen de pas (succes)', fontsize=12)
    for i, v in enumerate(avg_steps):
        axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)

    # 3. Temps moyen de recherche
    avg_times = []
    for algo in algo_names:
        times = [r.get('avg_search_time', 0) for r in results[algo]]
        avg_times.append(np.mean(times))

    axes[2].bar(algo_names, avg_times, color=colors)
    axes[2].set_title('Temps moyen de recherche (s)', fontsize=12)
    for i, v in enumerate(avg_times):
        axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    plt.suptitle('Comparaison des algorithmes MCTS pour la planification de trajectoire',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_comparison(env: GridEnvironment,
                                trajectories: Dict[str, List[Tuple[int, int]]],
                                save_path: str = None, show: bool = True):
    """
    Affiche les trajectoires de differents algorithmes sur la meme grille.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    grid = env.grid.copy()
    gx, gy = env.goal_pos
    grid[gx, gy] = CELL_GOAL

    ax.imshow(grid, cmap=CUSTOM_CMAP, vmin=0, vmax=4, origin='upper')

    colors_traj = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    for i, (algo_name, path) in enumerate(trajectories.items()):
        if path and len(path) > 1:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            color = colors_traj[i % len(colors_traj)]
            ax.plot(path_x, path_y, '-', linewidth=2.5, alpha=0.7,
                     color=color, label=f'{algo_name} ({len(path)} pas)')

    ax.legend(fontsize=11, loc='upper right')
    ax.set_title('Comparaison des trajectoires', fontsize=14)
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_search_time_evolution(search_times: Dict[str, List[float]],
                                save_path: str = None, show: bool = True):
    """
    Affiche l'evolution du temps de recherche au cours de la trajectoire.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for algo_name, times in search_times.items():
        ax.plot(times, label=algo_name, alpha=0.8)

    ax.set_xlabel('Pas de temps', fontsize=12)
    ax.set_ylabel('Temps de recherche (s)', fontsize=12)
    ax.set_title('Evolution du temps de recherche MCTS', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
