import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder
from typing import List, Tuple, Optional


class Pathfinder:
    """A* pathfinding on CSpace grid using pathfinding library."""

    def __init__(self, cspace):
        self.cspace = cspace
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

    def find(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
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

    def visualize(self, path: List[Tuple[float, float]], ax=None, title: str = "Path"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

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
