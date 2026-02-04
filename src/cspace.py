import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import Tuple


class CSpace:
    """2D Configuration space (x, y) for omnidirectional robots."""

    def __init__(self, warehouse, robot, resolution: float = 0.5):
        self.warehouse = warehouse
        self.robot = robot
        self.resolution = resolution
        self.cols = int(warehouse.width / resolution)
        self.rows = int(warehouse.height / resolution)
        # 2D grid: True = occupied, False = free
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)
        self._compute()

    def _compute(self):
        """Compute collision grid for all (x, y) configurations."""
        obstacles = self.warehouse.get_polygons()

        for row in range(self.rows):
            for col in range(self.cols):
                # Cell center in world coordinates
                x = (col + 0.5) * self.resolution
                y = (row + 0.5) * self.resolution
                robot_poly = self.robot.at(x, y)

                # Check boundary collision
                if not self._inside_bounds(robot_poly):
                    self.grid[row, col] = True
                    continue

                # Check obstacle collision
                for obs in obstacles:
                    if robot_poly.intersects(obs):
                        self.grid[row, col] = True
                        break

    def _inside_bounds(self, polygon: Polygon) -> bool:
        """Check if polygon is fully inside warehouse bounds."""
        minx, miny, maxx, maxy = polygon.bounds
        return minx >= 0 and miny >= 0 and maxx <= self.warehouse.width and maxy <= self.warehouse.height

    def is_free(self, x: float, y: float) -> bool:
        """Check if configuration (x, y) is collision-free."""
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return not self.grid[row, col]
        return False

    def to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        return int(y / self.resolution), int(x / self.resolution)

    def to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        return (col + 0.5) * self.resolution, (row + 0.5) * self.resolution

    def visualize(self, ax=None, title: str = "Configuration Space"):
        """Visualize C-Space with obstacles inflated by robot geometry."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Show collision grid (gray = occupied)
        extent = [0, self.warehouse.width, 0, self.warehouse.height]
        ax.imshow(self.grid, origin='lower', extent=extent, cmap='Greys', alpha=0.7)

        # Overlay original obstacles
        for polygon, vertices, name in self.warehouse.obstacles:
            ax.fill(*zip(*vertices), color='#8B4513', edgecolor='black')

        ax.set_xlim(0, self.warehouse.width)
        ax.set_ylim(0, self.warehouse.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    def save(self, filename: str, dpi: int = 150):
        """Save visualization to file."""
        fig, ax = plt.subplots(figsize=(10, 8))
        self.visualize(ax)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
