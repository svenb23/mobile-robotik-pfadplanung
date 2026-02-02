import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from typing import List, Tuple


class Robot:
    """Robot geometry as polygon."""

    def __init__(self, vertices: List[Tuple[float, float]]):
        self.vertices = vertices
        self.polygon = Polygon(vertices)

    def at(self, x: float, y: float) -> Polygon:
        translated = [(vx + x, vy + y) for vx, vy in self.vertices]
        return Polygon(translated)

    def visualize(self, ax=None, position: Tuple[float, float] = (0, 0)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        translated = [(vx + position[0], vy + position[1]) for vx, vy in self.vertices]
        patch = MplPolygon(translated, facecolor='#2E86AB', edgecolor='black', alpha=0.8)
        ax.add_patch(patch)
        ax.plot(position[0], position[1], 'ko', markersize=4)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return ax

    @classmethod
    def circle(cls, radius: float, n_points: int = 16) -> 'Robot':
        """Creates a circular robot approximated by polygon."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        vertices = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        return cls(vertices)

    @classmethod
    def rectangle(cls, width: float, height: float) -> 'Robot':
        """Creates a rectangular robot centered at origin."""
        w, h = width / 2, height / 2
        return cls([(-w, -h), (w, -h), (w, h), (-w, h)])
