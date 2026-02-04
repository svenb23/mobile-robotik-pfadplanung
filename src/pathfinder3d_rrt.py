import numpy as np
from typing import List, Tuple, Optional
import random


class Pathfinder3D_RRT:
    """RRT (Rapidly-exploring Random Tree) in 3D configuration space."""

    def __init__(self, cspace3d):
        self.cspace = cspace3d

    def find(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             max_iter: int = 10000, step_size: float = 0.5,
             angle_step: float = 0.3) -> Optional[List[Tuple[float, float, float]]]:
        """Find path using RRT algorithm."""
        tree = {start: None}

        for _ in range(max_iter):
            # Sample random point (10% bias towards goal)
            if random.random() < 0.1:
                sample = goal
            else:
                sample = (
                    random.uniform(0, self.cspace.warehouse.width),
                    random.uniform(0, self.cspace.warehouse.height),
                    random.uniform(0, 2 * np.pi)
                )

            # Find nearest node in tree
            nearest = min(tree.keys(), key=lambda n: self._dist(n, sample))

            # Steer towards sample
            direction = np.array(sample[:2]) - np.array(nearest[:2])
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                new_x, new_y = nearest[0], nearest[1]
            else:
                direction = direction / dist
                new_pos = np.array(nearest[:2]) + direction * min(step_size, dist)
                new_x, new_y = new_pos[0], new_pos[1]

            # Interpolate angle
            angle_diff = sample[2] - nearest[2]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            new_theta = nearest[2] + np.sign(angle_diff) * min(abs(angle_diff), angle_step)
            new_theta = new_theta % (2 * np.pi)

            new_point = (new_x, new_y, new_theta)

            # Add to tree if path is collision-free
            if self._line_free(nearest, new_point):
                tree[new_point] = nearest

                # Check if goal is reachable
                if self._dist(new_point, goal) < step_size:
                    if self._line_free(new_point, goal):
                        tree[goal] = new_point
                        return self._reconstruct(tree, goal)

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

    def _reconstruct(self, tree, goal):
        """Reconstruct path from tree."""
        path = [goal]
        current = goal
        while tree[current] is not None:
            current = tree[current]
            path.append(current)
        return path[::-1]
