import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import Tuple


class CSpace3D:
    """3D Configuration space (x, y, theta) for non-holonomic robots."""

    def __init__(self, warehouse, robot, resolution: float = 0.5, n_angles: int = 12):
        self.warehouse = warehouse
        self.robot = robot
        self.resolution = resolution
        self.n_angles = n_angles
        self.angle_resolution = 2 * np.pi / n_angles
        self.cols = int(warehouse.width / resolution)
        self.rows = int(warehouse.height / resolution)
        # 3D grid: True = occupied, False = free
        self.grid = np.zeros((self.rows, self.cols, n_angles), dtype=bool)
        self._compute()

    def _compute(self):
        """Compute collision grid for all (x, y, theta) configurations."""
        obstacles = self.warehouse.get_polygons()
        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        for row in range(self.rows):
            for col in range(self.cols):
                x = (col + 0.5) * self.resolution
                y = (row + 0.5) * self.resolution
                for k, theta in enumerate(angles):
                    robot_poly = self.robot.at(x, y, theta)
                    # Check boundary collision
                    if not self._inside_bounds(robot_poly):
                        self.grid[row, col, k] = True
                        continue
                    # Check obstacle collision
                    for obs in obstacles:
                        if robot_poly.intersects(obs):
                            self.grid[row, col, k] = True
                            break

    def _inside_bounds(self, polygon: Polygon) -> bool:
        minx, miny, maxx, maxy = polygon.bounds
        return minx >= 0 and miny >= 0 and maxx <= self.warehouse.width and maxy <= self.warehouse.height

    def is_free(self, x: float, y: float, theta: float) -> bool:
        """Check if configuration (x, y, theta) is collision-free."""
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        k = int((theta % (2 * np.pi)) / self.angle_resolution) % self.n_angles
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return not self.grid[row, col, k]
        return False

    def to_grid(self, x: float, y: float, theta: float) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        row = int(y / self.resolution)
        col = int(x / self.resolution)
        k = int((theta % (2 * np.pi)) / self.angle_resolution) % self.n_angles
        return row, col, k

    def to_world(self, row: int, col: int, k: int) -> Tuple[float, float, float]:
        """Convert grid indices to world coordinates."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        theta = k * self.angle_resolution
        return x, y, theta

    def visualize_3d(self, path=None, ax=None, title: str = "3D C-Space"):
        """Visualize C-Space with obstacles as 3D pillars."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

        # Draw obstacles as vertical walls extending through all angles
        for polygon, vertices, name in self.warehouse.obstacles:
            verts = list(vertices)
            for i in range(len(verts)):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % len(verts)]
                wall = [[x1, y1, 0], [x2, y2, 0], [x2, y2, 360], [x1, y1, 360]]
                ax.add_collection3d(Poly3DCollection([wall], alpha=0.3, facecolor='#8B4513', edgecolor='#5D3A1A'))

        # Draw path if provided
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            path_theta = [np.degrees(p[2]) for p in path]
            ax.plot(path_x, path_y, path_theta, 'b-', linewidth=3, label='Path')
            ax.scatter([path_x[0]], [path_y[0]], [path_theta[0]], c='green', s=150, marker='o', label='Start')
            ax.scatter([path_x[-1]], [path_y[-1]], [path_theta[-1]], c='red', s=150, marker='o', label='Goal')
            ax.legend()

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('θ [°]')
        ax.set_title(title)
        ax.set_xlim(0, self.warehouse.width)
        ax.set_ylim(0, self.warehouse.height)
        ax.set_zlim(0, 360)
        return ax
