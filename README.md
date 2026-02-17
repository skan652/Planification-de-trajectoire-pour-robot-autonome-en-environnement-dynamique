# Planification de Trajectoire pour Robot Autonome en Environnement Dynamique

Implémentation complète d'algorithmes de recherche arborescente Monte-Carlo (MCTS) pour la planification de trajectoire de robots autonomes dans des environnements dynamiques avec obstacles mobiles.

## 📋 Présentation

Ce projet implémente et compare plusieurs variantes de MCTS de pointe pour la navigation de robots dans des environnements de grille 2D avec obstacles statiques et dynamiques. Le robot doit planifier des chemins sûrs et efficaces pour atteindre un objectif tout en évitant les collisions avec des obstacles qui peuvent se déplacer de manière imprévisible.

## ✨ Fonctionnalités

- **Algorithmes MCTS multiples :**
  - Flat Monte Carlo (référence)
  - MCTS avec UCT (Upper Confidence bounds for Trees)
  - MCTS avec RAVE (Rapid Action Value Estimation)
  - MCTS avec GRAVE (Generalized RAVE)

- **Gestion d'environnements dynamiques :**
  - Replanification en temps réel lors de changements d'environnement
  - Planificateur adaptatif pour prise de décision en ligne
  - Support des obstacles mobiles avec modèles de vélocité

- **Tests complets :**
  - Plusieurs types de scénarios (simple, complexe, passages étroits)
  - Simulation de bruit de capteurs
  - Comparaison statistique entre algorithmes

- **Outils de visualisation :**
  - Visualisation de trajectoires
  - Exploration de l'arbre de recherche
  - Comparaison de performances d'algorithmes
  - Analyse d'évolution temporelle

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- Gestionnaire de paquets pip

### Configuration

**1. Cloner le dépôt :**

```bash
git clone https://github.com/keita223/Planification-de-trajectoire-pour-robot-autonome-en-environnement-dynamique.git
cd Planification-de-trajectoire-pour-robot-autonome-en-environnement-dynamique
```

**2. Installer les dépendances :**

```bash
pip install -r requirements.txt
```

Les paquets requis sont :

- `numpy>=1.24.0` - Calculs numériques
- `matplotlib>=3.7.0` - Visualisation
- `pygame>=2.5.0` - Visualisation interactive
- `tqdm>=4.65.0` - Barres de progression
- `scipy>=1.10.0` - Calcul scientifique

## 📖 Utilisation

### Lancer les expérimentations

Exécuter la suite complète d'expériences :

```bash
python tests/run_experiments.py
```

Cela va :

- Comparer différents algorithmes MCTS (Flat MC, UCT, RAVE, GRAVE)
- Tester divers scénarios (statique, dynamique, passages étroits)
- Générer des métriques de performance et visualisations dans le répertoire `results/`

### Lancer des tests individuels

Tester l'environnement :

```bash
python tests/test_environment.py
```

Tester les algorithmes MCTS :

```bash
python tests/test_mcts.py
```

### Utiliser dans votre code

```python
from src.environment import GridEnvironment, create_simple_scenario
from src.mcts_base import MCTS_UCT
from src.visualization import plot_trajectory_comparison

# Créer l'environnement
env = create_simple_scenario(grid_size=20)

# Initialiser le planificateur MCTS
planner = MCTS_UCT(env, n_simulations=1000, exploration_constant=1.414)

# Planifier le chemin
path = planner.plan()

# Visualiser
plot_trajectory_comparison(env, path, save_path="mon_chemin.png")
```

## 📁 Structure du projet

```text
.
├── src/
│   ├── __init__.py
│   ├── environment.py          # Environnement de grille avec obstacles
│   ├── mcts_base.py            # Implémentations Flat MC et UCT
│   ├── mcts_rave.py            # Implémentations RAVE et GRAVE
│   ├── mcts_dynamic.py         # Algorithmes de replanification dynamique
│   └── visualization.py        # Outils de visualisation et de graphiques
├── tests/
│   ├── __init__.py
│   ├── run_experiments.py      # Suite expérimentale principale
│   ├── test_environment.py     # Tests unitaires de l'environnement
│   └── test_mcts.py            # Tests des algorithmes MCTS
├── rapport/
│   └── rapport.tex             # Rapport de recherche LaTeX
├── results/                    # Résultats et graphiques générés
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🎯 Algorithmes implémentés

### 1. Flat Monte Carlo (référence)

Échantillonnage aléatoire pur sans recherche arborescente. Utile comme référence pour la comparaison.

### 2. MCTS avec UCT

Recherche arborescente Monte-Carlo standard utilisant la formule UCT (Upper Confidence bounds for Trees) pour équilibrer exploration et exploitation :

$$UCT = \frac{W_i}{N_i} + C \sqrt{\frac{\ln N_p}{N_i}}$$

où :

- $W_i$ = récompense totale pour l'action $i$
- $N_i$ = nombre de fois que l'action $i$ a été visitée
- $N_p$ = nombre de fois que le parent a été visité
- $C$ = constante d'exploration

### 3. MCTS avec RAVE

Rapid Action Value Estimation (RAVE) utilise les statistiques AMAF (All Moves As First) pour accélérer l'apprentissage. Il combine les valeurs UCT et AMAF :

$$RAVE = (1-\beta) \cdot UCT + \beta \cdot AMAF$$

### 4. MCTS avec GRAVE

Generalized RAVE utilise les statistiques AMAF des ancêtres pour une estimation de valeur plus fiable dans les premières phases de recherche.

### 5. Planificateur dynamique

Système de replanification en temps réel qui :

- Exécute une action à la fois
- Observe les changements d'environnement
- Replanifie si nécessaire
- Maintient un état de croyance sur les positions d'obstacles

### 6. Planificateur adaptatif

Planificateur dynamique amélioré avec :

- Budget de simulation ajustable selon la complexité de l'environnement
- Réponse rapide aux situations urgentes
- Planification complète quand le temps le permet

## 📊 Scénarios

Le projet inclut plusieurs scénarios de test :

- **Scénario simple :** Recherche de chemin basique avec peu d'obstacles statiques
- **Scénario dynamique :** Obstacles mobiles avec vélocités définies
- **Scénario complexe :** Champ d'obstacles dense nécessitant une navigation prudente
- **Passages étroits :** Teste la capacité à trouver des chemins dans des espaces restreints
- **Bruit de capteurs :** Simule une détection d'obstacles imparfaite

## 🎨 Visualisation

Le projet génère diverses visualisations :

- **Graphiques d'environnement :** Affichent la grille, les obstacles, le robot et l'objectif
- **Comparaisons de trajectoires :** Comparent les chemins de différents algorithmes
- **Métriques de performance :** Taux de succès, longueur de chemin, temps de calcul
- **Visualisation de l'arbre de recherche :** Explore le processus de décision MCTS
- **Évolution temporelle :** Suit l'efficacité de planification au fil des itérations

---

*Pour la documentation technique détaillée, consultez le rapport LaTeX dans le répertoire `rapport/`.*
