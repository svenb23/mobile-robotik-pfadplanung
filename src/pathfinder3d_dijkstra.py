import numpy as np
from typing import List, Tuple, Optional
import heapq


class Pathfinder3D_Dijkstra:
    """Dijkstra pathfinding in 3D configuration space."""

    def __init__(self, cspace3d):
        self.cspace = cspace3d

    def find(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """Find optimal path from start to goal using Dijkstra (no heuristic)."""
        start_cell = self.cspace.to_grid(*start)
        goal_cell = self.cspace.to_grid(*goal)

        if self.cspace.grid[start_cell] or self.cspace.grid[goal_cell]:
            return None

        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}

        # 8-connectivity in xy + 3 options for theta
        neighbors_xy = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)]
        neighbors_theta = [-1, 0, 1]

        while open_set:
            current_g, current = heapq.heappop(open_set)

            if current == goal_cell:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return [self.cspace.to_world(*cell) for cell in path[::-1]]

            if current_g > g_score.get(current, float('inf')):
                continue

            row, col, k = current

            for dr, dc in neighbors_xy:
                for dk in neighbors_theta:
                    if dr == 0 and dc == 0 and dk == 0:
                        continue

                    nr = row + dr
                    nc = col + dc
                    nk = (k + dk) % self.cspace.n_angles

                    if not (0 <= nr < self.cspace.rows and 0 <= nc < self.cspace.cols):
                        continue
                    if self.cspace.grid[nr, nc, nk]:
                        continue

                    neighbor = (nr, nc, nk)
                    # Cost = spatial distance + rotation cost
                    spatial_dist = np.sqrt(dr**2 + dc**2) * self.cspace.resolution
                    rotation_cost = abs(dk) * self.cspace.angle_resolution * 0.5
                    tentative_g = g_score[current] + spatial_dist + rotation_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g, neighbor))

        return None
