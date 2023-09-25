"""
Ce fichier contient des exercices à compléter sur la programmation dynamique.
Il est évalué automatiquement avec pytest, vous pouvez le lancer avec la
commande `pytest exercices.py`.
"""
import typing as t
import random
import pytest
import numpy as np
import gym
from gym import spaces


"""
Partie 1 - Processus décisionnels de Markov
===========================================

Rappel: un processus décisionnel de Markov (MDP) est un modèle de
décision séquentielle dans lequel les états du système sont décrits par
une variable aléatoire, et les actions du système sont décrites par une
autre variable aléatoire. Les transitions entre états sont décrites par
une matrice de transition, et les récompenses associées à chaque
transition sont décrites par une matrice de récompenses.

Dans ce TP, nous allons utiliser la librairie `gym` pour implémenter un
MDP simple, et nous allons utiliser la programmation dynamique pour
résoudre ce MDP.
"""

# Exercice 1: MDP simple
# ----------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP doit avoir
# 3 états, 2 actions, et les transitions et récompenses suivantes:
#   - état 0, action 0 -> état 1, récompense -1
#   - état 0, action 1 -> état 0, récompense -1
#   - état 1, action 0 -> état 0, récompense -1
#   - état 1, action 1 -> état 2, récompense -1
#   - état 2, action 0 -> état 2, récompense 0
#   - état 2, action 1 -> état 0, récompense -1


class MDP(gym.Env):
    """
    MDP simple avec 3 états et 2 actions.
    """

    observation_space: spaces.Discrete
    action_space: spaces.Discrete

    # state, action -> [(next_state, reward, done)]
    P: list[list[tuple[int, float, bool]]]

    def __init__(self):
        # BEGIN SOLUTION
        # END SOLUTION

    def step(self, action: int) -> tuple[int, float, bool, dict]:  # type: ignore
        """
        Effectue une transition dans le MDP.
        Renvoie l'observation suivante, la récompense, un booléen indiquant
        si l'épisode est terminé, et un dictionnaire d'informations.
        """
        # BEGIN SOLUTION
        # END SOLUTION


# Tests pour l'exercice 1
def test_mdp():
    mdp = MDP()
    assert mdp.P[0][0] == (1, -1, False)
    assert mdp.P[0][1] == (0, -1, False)
    assert mdp.P[1][0] == (0, -1, False)
    assert mdp.P[1][1] == (2, -1, False)
    assert mdp.P[2][0] == (2, 0, False)
    assert mdp.P[2][1] == (0, -1, False)

    mdp.reset()
    ret = mdp.step(0)
    assert ret[0] in [0, 1, 2]
    assert ret[1] in [0, 1]
    assert ret[2] in [True, False]
    assert isinstance(ret[3], dict)


# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    # END SOLUTION
    return values


def test_mdp_value_iteration():
    mdp = MDP()
    values = mdp_value_iteration(mdp, max_iter=1000, gamma=1.0)
    assert np.allclose(values, [-2, -1, 0])
    values = mdp_value_iteration(mdp, max_iter=1000, gamma=0.9)
    assert np.allclose(values, [-1.9, -1, 0])


# Exercice 3: Extension du MDP à un GridWorld (sans bruit)
# --------------------------------------------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP est formé
# d'un GridWorld de 3x4 cases, avec 4 actions possibles (haut, bas, gauche,
# droite). La case (1, 1) est inaccessible (mur), tandis que la case (1, 3)
# est un état terminal avec une récompense de -1. La case (0, 3) est un état
# terminal avec une récompense de +1. Tout autre état a une récompense de 0.
# L'agent commence dans la case (0, 0).

# Complétez la classe ci-dessous pour implémenter ce MDP.
# Puis, utilisez l'algorithme de value iteration pour calculer la fonction de
# valeur de chaque état.


class GridWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    # F: Free, S: Start, P: Positive reward, N: negative reward, W: Wall
    grid: np.ndarray = np.array(
        [
            ["F", "F", "F", "P"],
            ["F", "W", "F", "N"],
            ["F", "F", "F", "F"],
            ["S", "F", "F", "F"],
        ]
    )
    current_position: tuple[int, int] = (0, 0)

    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))

        self.current_position = (0, 0)

    def step(self, action):
        if action == 0:  # Up
            self.current_position = (
                max(0, self.current_position[0] - 1),
                self.current_position[1],
            )
        elif action == 1:  # Down
            self.current_position = (
                min(3, self.current_position[0] + 1),
                self.current_position[1],
            )
        elif action == 2:  # Left
            self.current_position = (
                self.current_position[0],
                max(0, self.current_position[1] - 1),
            )
        elif action == 3:  # Right
            self.current_position = (
                self.current_position[0],
                min(3, self.current_position[1] + 1),
            )

        next_state = tuple(self.current_position)

        # Check if the agent has reached the goal
        is_done = self.grid[tuple(self.current_position)] in {"P", "N"}

        # Provide reward
        if self.grid[tuple(self.current_position)] == "N":
            reward = -1
        elif self.grid[tuple(self.current_position)] == "P":
            reward = 1
        else:
            reward = 0

        return next_state, reward, is_done, {}

    def reset(self):
        self.current_position = (0, 0)  # Start Position
        return self.current_position

    def render(self):
        for row in range(4):
            for col in range(4):
                if self.current_position == [row, col]:
                    print("X", end=" ")
                else:
                    print(self.grid[row, col], end=" ")
            print("")  # Newline at the end of the row


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    # END SOLUTION


def test_grid_world_value_iteration():
    env = GridWorldEnv()

    values = grid_world_value_iteration(env, max_iter=1000, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution)

    values = grid_world_value_iteration(env, max_iter=1000, gamma=0.9)
    solution = np.array(
        [
            [0.81, 0.9, 1.0, 0.0],
            [0.729, 0.0, 0.9, 0.0],
            [0.6561, 0.729, 0.81, 0.729],
            [0.59049, 0.6561, 0.729, 0.6561],
        ]
    )
    assert np.allclose(values, solution)


# Exercice 4: GridWorld avec du bruit
# -----------------------------------
# Ecrire une fonction qui calcule la fonction de valeur pour le GridWorld
# avec du bruit.
# Le bruit est un mouvement aléatoire de l'agent vers sa gauche ou sa droite avec une probabilité de 0.1.


class StochasticGridWorldEnv(GridWorldEnv):
    def _add_noise(self, action: int) -> int:
        prob = random.uniform(0, 1)
        if prob < 0.05:  # 10% chance to go left
            return (action - 1) % 4
        elif prob < 0.1:  # 10% chance to go right
            return (action + 1) % 4
        # 80% chance to go in the intended direction
        return action

    def step(self, action):
        action = self._add_noise(action)
        return super().step(action)


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    # END SOLUTION
    return values


def test_stochastic_grid_world_value_iteration():
    env = StochasticGridWorldEnv()

    values = stochastic_grid_world_value_iteration(env, max_iter=1000, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution)

    values = stochastic_grid_world_value_iteration(env, max_iter=1000, gamma=0.9)
    solution = np.array(
        [
            [0.73150697, 0.83310665, 0.96151603, 0.0],
            [0.63232473, 0.0, 0.64154523, 0.0],
            [0.54146311, 0.48655038, 0.54726419, 0.47417735],
            [0.47112049, 0.43185906, 0.47417735, 0.41635033],
        ]
    )
    assert np.allclose(values, solution)


# Exercice 3: Evaluation de politique
# -----------------------------------
# Ecrire une fonction qui évalue la politique suivante:
#   - état 0, action 0
#   - état 1, action 0
#   - état 2, action 1


"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    # BEGIN SOLUTION
    # END SOLUTION


# Tests pour l'exercice 1
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """
    # BEGIN SOLUTION
    # END SOLUTION


# Tests pour l'exercice 2
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci_memo(n, expected):
    assert fibonacci_memo(n) == expected


# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    # END SOLUTION


# Tests pour l'exercice 3
@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 0),
        (2, 3),
        (3, 0),
        (4, 11),
        (5, 0),
        (6, 41),
        (7, 0),
        (8, 153),
        (9, 0),
        (10, 571),
    ],
)
def test_domino_paving(n, expected):
    assert domino_paving(n) == expected
