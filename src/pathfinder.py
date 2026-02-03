import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.best_first import BestFirst
from typing import List, Tuple, Optional
import random


ALGORITHMS = ["a_star", "dijkstra", "best_first", "rrt", "prm"]


class Pathfinder:
    """Pathfinding on CSpace grid with multiple algorithms."""

    def __init__(self, cspace, algorithm: str = "a_star"):
        self.cspace = cspace
        self.algorithm = algorithm
        self._init_finder()

    def _init_finder(self):
        if self.algorithm == "a_star":
            self.finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        elif self.algorithm == "dijkstra":
            self.finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
        elif self.algorithm == "best_first":
            self.finder = BestFirst(diagonal_movement=DiagonalMovement.always)
        elif self.algorithm in ["rrt", "prm"]:
            self.finder = None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def find(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        if self.algorithm in ["a_star", "dijkstra", "best_first"]:
            return self._find_grid(start, goal)
        elif self.algorithm == "rrt":
            return self._find_rrt(start, goal)
        elif self.algorithm == "prm":
            return self._find_prm(start, goal)

    def _find_grid(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_cell = self.cspace.to_grid(start[0], start[1])
        goal_cell = self.cspace.to_grid(goal[0], goal[1])

        matrix = (~self.cspace.grid).astype(int).tolist()
        grid = Grid(matrix=matrix)

        start_node = grid.node(start_cell[1], start_cell[0])
        goal_node = grid.node(goal_cell[1], goal_cell[0])

        path, _ = self.finder.find_path(start_node, goal_node, grid)

        if not path:
            return None

        return [self.cspace.to_world(row, col) for col, row in path]

    def _find_rrt(self, start: Tuple[float, float], goal: Tuple[float, float],
                  max_iter: int = 5000, step_size: float = 0.5) -> Optional[List[Tuple[float, float]]]:
        """Rapidly-exploring Random Tree algorithm."""
        tree = {start: None}

        for _ in range(max_iter):
            if random.random() < 0.1:
                sample = goal
            else:
                sample = (
                    random.uniform(0, self.cspace.warehouse.width),
                    random.uniform(0, self.cspace.warehouse.height)
                )

            nearest = min(tree.keys(), key=lambda n: self._dist(n, sample))

            direction = np.array(sample) - np.array(nearest)
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction = direction / dist
            new_point = tuple(np.array(nearest) + direction * min(step_size, dist))

            if self._line_free(nearest, new_point):
                tree[new_point] = nearest

                if self._dist(new_point, goal) < step_size and self._line_free(new_point, goal):
                    tree[goal] = new_point
                    return self._reconstruct_path(tree, goal)

        return None

    def _find_prm(self, start: Tuple[float, float], goal: Tuple[float, float],
                  n_samples: int = 500, k_neighbors: int = 10) -> Optional[List[Tuple[float, float]]]:
        samples = [start, goal]
        for _ in range(n_samples):
            point = (
                random.uniform(0, self.cspace.warehouse.width),
                random.uniform(0, self.cspace.warehouse.height)
            )
            if self.cspace.is_free(point[0], point[1]):
                samples.append(point)

        edges = {s: [] for s in samples}
        for s in samples:
            neighbors = sorted(samples, key=lambda n: self._dist(s, n))[1:k_neighbors+1]
            for n in neighbors:
                if self._line_free(s, n):
                    edges[s].append(n)
                    edges[n].append(s)

        return self._astar_roadmap(start, goal, edges)

    def _astar_roadmap(self, start, goal, edges) -> Optional[List[Tuple[float, float]]]:
        import heapq
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in edges.get(current, []):
                tentative_g = g_score[current] + self._dist(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._dist(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _line_free(self, a: Tuple[float, float], b: Tuple[float, float], steps: int = 20) -> bool:
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            if not self.cspace.is_free(x, y):
                return False
        return True

    def _reconstruct_path(self, tree: dict, goal) -> List[Tuple[float, float]]:
        path = [goal]
        current = goal
        while tree[current] is not None:
            current = tree[current]
            path.append(current)
        return path[::-1]

    def visualize(self, path: List[Tuple[float, float]], ax=None, title: str = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if title is None:
            title = f"Path ({self.algorithm})"

        self.cspace.visualize(ax, title=title)

        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, 'b-', linewidth=2, label='Path')
            ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
            ax.plot(xs[-1], ys[-1], 'ro', markersize=10, label='Goal')
            ax.legend()

        return ax

    def save(self, path: List[Tuple[float, float]], filename: str, dpi: int = 150):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.visualize(path, ax)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
