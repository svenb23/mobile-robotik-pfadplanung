import numpy as np
from typing import List, Tuple, Optional
import random
import heapq


class Pathfinder3D_PRM:
    """PRM (Probabilistic Roadmap) in 3D configuration space."""

    def __init__(self, cspace3d):
        self.cspace = cspace3d

    def find(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             n_samples: int = 1000, k_neighbors: int = 15) -> Optional[List[Tuple[float, float, float]]]:
        """Find path using PRM algorithm."""
        # Sample random collision-free configurations
        samples = [start, goal]
        for _ in range(n_samples):
            point = (
                random.uniform(0, self.cspace.warehouse.width),
                random.uniform(0, self.cspace.warehouse.height),
                random.uniform(0, 2 * np.pi)
            )
            if self.cspace.is_free(*point):
                samples.append(point)

        # Build roadmap by connecting k-nearest neighbors
        edges = {s: [] for s in samples}
        for s in samples:
            neighbors = sorted(samples, key=lambda n: self._dist(s, n))[1:k_neighbors+1]
            for n in neighbors:
                if self._line_free(s, n):
                    edges[s].append(n)
                    edges[n].append(s)

        # Search roadmap using A*
        return self._astar(start, goal, edges)

    def _astar(self, start, goal, edges):
        """A* search on the roadmap."""
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

    def _dist(self, a, b):
        """Distance metric combining spatial and angular distance."""
        spatial = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        angle_diff = abs(a[2] - b[2])
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        return spatial + angle_diff * 0.5

    def _line_free(self, a, b, steps: int = 10):
        """Check if straight line between two configurations is collision-free."""
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            angle_diff = b[2] - a[2]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            theta = (a[2] + t * angle_diff) % (2 * np.pi)
            if not self.cspace.is_free(x, y, theta):
                return False
        return True
