import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point
from typing import List, Tuple


class Warehouse:
    """Warehouse environment with polygon obstacles."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.obstacles: List[Tuple[Polygon, List, str]] = []

    def add(self, vertices: List[Tuple[float, float]], name: str = "") -> None:
        """Add polygon obstacle defined by vertices."""
        polygon = Polygon(vertices)
        self.obstacles.append((polygon, vertices, name))

    def is_free(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside bounds and not colliding with obstacles."""
        x, y = point
        if not (0 < x < self.width and 0 < y < self.height):
            return False
        for polygon, _, _ in self.obstacles:
            if polygon.contains(Point(point)):
                return False
        return True

    def get_polygons(self) -> List[Polygon]:
        """Return list of obstacle polygons."""
        return [polygon for polygon, _, _ in self.obstacles]

    def visualize(self, ax=None, title: str = "Warehouse"):
        """Visualize warehouse with obstacles."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Draw border
        ax.plot([0, self.width, self.width, 0, 0],
                [0, 0, self.height, self.height, 0], 'k-', linewidth=2)

        # Draw obstacles
        for _, vertices, name in self.obstacles:
            patch = MplPolygon(vertices, facecolor='#8B4513', edgecolor='black')
            ax.add_patch(patch)
            if name:
                bounds = Polygon(vertices).bounds
                cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
                ax.text(cx, cy, name, ha='center', va='center', fontsize=7, color='white')

        ax.set_xlim(-1, self.width + 1)
        ax.set_ylim(-1, self.height + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    def save(self, filename: str, dpi: int = 150) -> None:
        """Save visualization to file."""
        fig, ax = plt.subplots(figsize=(10, 8))
        self.visualize(ax)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
