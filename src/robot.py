import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple


class Robot:
    """Robot geometry defined by polygon vertices centered at origin."""

    def __init__(self, vertices: List[Tuple[float, float]]):
        self.vertices = vertices
        self.polygon = Polygon(vertices)

    def at(self, x: float, y: float, theta: float = 0) -> Polygon:
        """Return robot polygon at position (x,y) with rotation theta."""
        if theta == 0:
            transformed = [(vx + x, vy + y) for vx, vy in self.vertices]
        else:
            # Rotation matrix: [cos -sin; sin cos]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            transformed = [
                (vx * cos_t - vy * sin_t + x, vx * sin_t + vy * cos_t + y)
                for vx, vy in self.vertices
            ]
        return Polygon(transformed)

    @classmethod
    def rectangle(cls, width: float, height: float) -> 'Robot':
        """Create rectangular robot centered at origin."""
        w, h = width / 2, height / 2
        return cls([(-w, -h), (w, -h), (w, h), (-w, h)])

    @classmethod
    def circle(cls, radius: float, n_points: int = 16) -> 'Robot':
        """Create circular robot centered at origin (approximated as polygon)."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        vertices = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        return cls(vertices)

    @classmethod
    def triangle(cls, base: float, height: float) -> 'Robot':
        """Create triangular robot centered at origin (pointing forward)."""
        return cls([
            (height / 2, 0),          
            (-height / 2, -base / 2),  
            (-height / 2, base / 2),   
        ])
