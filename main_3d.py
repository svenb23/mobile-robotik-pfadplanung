import time
import numpy as np
import matplotlib.pyplot as plt
from src import Warehouse, Robot, CSpace3D, MAPS
from src.pathfinder3d import Pathfinder3D as AStar3D
from src.pathfinder3d_dijkstra import Pathfinder3D_Dijkstra as Dijkstra3D
from src.pathfinder3d_rrt import Pathfinder3D_RRT as RRT3D
from src.pathfinder3d_prm import Pathfinder3D_PRM as PRM3D

RESOLUTION = 0.5
N_ANGLES = 12
ROBOT = Robot.rectangle(1.0, 0.5)

map_data = MAPS["hard"]
warehouse = Warehouse(map_data["width"], map_data["height"])
for obs in map_data["obstacles"]:
    warehouse.add(obs)

start = (1, 1, 0)
goal = (7, 2, np.pi)

t0 = time.time()
cspace = CSpace3D(warehouse, ROBOT, resolution=RESOLUTION, n_angles=N_ANGLES)
t_cspace = time.time() - t0
print(f"Grid: {cspace.rows} x {cspace.cols} x {cspace.n_angles} = {cspace.rows * cspace.cols * cspace.n_angles} cells ({t_cspace:.2f}s)")

algorithms = [
    ("A*", AStar3D(cspace)),
    ("Dijkstra", Dijkstra3D(cspace)),
    ("RRT", RRT3D(cspace)),
    ("PRM", PRM3D(cspace)),
]

results = []
for name, pf in algorithms:
    t0 = time.time()
    path = pf.find(start, goal)
    t = time.time() - t0
    if path:
        length = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2) for i in range(len(path)-1))
        print(f"{name}: {len(path)} points, {length:.2f}m, {t:.3f}s")
        results.append((name, path, length, t))
    else:
        print(f"{name}: No path found")

fig = plt.figure(figsize=(16, 4))
for i, (name, path, length, t) in enumerate(results):
    ax = fig.add_subplot(1, len(results), i+1, projection='3d')
    cspace.visualize_3d(path, ax=ax, title=f"{name}\n{length:.1f}m, {t:.2f}s")
    ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig("figures/cspace3d_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figures/cspace3d_comparison.png")
