import time
import matplotlib.pyplot as plt
from src import Warehouse, Robot, CSpace, Pathfinder, MAPS

# Configuration
ALGORITHMS = ["a_star", "dijkstra", "best_first", "rrt", "prm"]
ROBOTS = {
    "circle": Robot.circle(0.5),
    "rectangle": Robot.rectangle(0.8, 0.5),
    "triangle": Robot([(0, -0.4), (0.5, 0.3), (-0.5, 0.3)]),
}
RESOLUTION = 0.3

# Start/Goal positions for each map
POSITIONS = {
    "easy": ((1, 1), (18, 13)),
    "medium": ((1, 1), (23, 18)),
    "hard": ((1, 1), (28, 23)),
}


def load_warehouse(map_name):
    map_data = MAPS[map_name]
    w = Warehouse(map_data["width"], map_data["height"])
    for obs in map_data["obstacles"]:
        w.add(obs)
    return w


def run_single(map_name, robot_name, algorithm):
    warehouse = load_warehouse(map_name)
    robot = ROBOTS[robot_name]
    start, goal = POSITIONS[map_name]

    cspace = CSpace(warehouse, robot, resolution=RESOLUTION)

    pathfinder = Pathfinder(cspace, algorithm=algorithm)

    t0 = time.time()
    path = pathfinder.find(start, goal)
    t1 = time.time()

    path_length = 0
    if path:
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += (dx**2 + dy**2) ** 0.5

    return {
        "map": map_name,
        "robot": robot_name,
        "algorithm": algorithm,
        "path_found": path is not None,
        "path_length": round(path_length, 2),
        "points": len(path) if path else 0,
        "time": round(t1 - t0, 3),
        "path": path,
        "cspace": cspace,
        "pathfinder": pathfinder,
    }


def run_all():
    results = []
    total = len(MAPS) * len(ROBOTS) * len(ALGORITHMS)
    count = 0

    for map_name in MAPS:
        for robot_name in ROBOTS:
            for algorithm in ALGORITHMS:
                count += 1
                result = run_single(map_name, robot_name, algorithm)
                results.append(result)

    return results


def save_results_table(results, filename="figures/results.txt"):
    with open(filename, "w") as f:
        f.write(f"{'Map':<10} {'Robot':<12} {'Algorithm':<12} {'Found':<6} {'Length':<10} {'Points':<8} {'Time':<8}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['map']:<10} {r['robot']:<12} {r['algorithm']:<12} "
                    f"{'Yes' if r['path_found'] else 'No':<6} {r['path_length']:<10} "
                    f"{r['points']:<8} {r['time']:<8}\n")


def save_comparison_figures(results):
    for map_name in MAPS:
        fig, axes = plt.subplots(len(ROBOTS), len(ALGORITHMS), figsize=(20, 12))

        for i, robot_name in enumerate(ROBOTS):
            for j, algorithm in enumerate(ALGORITHMS):
                ax = axes[i, j]

                result = next(r for r in results
                             if r["map"] == map_name
                             and r["robot"] == robot_name
                             and r["algorithm"] == algorithm)

                if result["path_found"]:
                    result["pathfinder"].visualize(result["path"], ax=ax,
                                                   title=f"{robot_name} | {algorithm}")
                else:
                    result["cspace"].visualize(ax=ax, title=f"{robot_name} | {algorithm} (No path)")

                if i == 0:
                    ax.set_title(f"{algorithm}\n{robot_name}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{robot_name}\nY [m]", fontsize=10)

        plt.suptitle(f"Map: {map_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"figures/comparison_{map_name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def save_algorithm_comparison(results):
    map_name = "medium"
    robot_name = "circle"

    fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(20, 4))

    for i, algorithm in enumerate(ALGORITHMS):
        result = next(r for r in results
                     if r["map"] == map_name
                     and r["robot"] == robot_name
                     and r["algorithm"] == algorithm)

        if result["path_found"]:
            result["pathfinder"].visualize(result["path"], ax=axes[i])
            axes[i].set_title(f"{algorithm}\nLength: {result['path_length']}m")
        else:
            result["cspace"].visualize(ax=axes[i], title=f"{algorithm}\nNo path")

    plt.suptitle(f"Algorithm Comparison (Map: {map_name}, Robot: {robot_name})", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_robot_comparison(results):
    map_name = "easy"
    algorithm = "a_star"

    fig, axes = plt.subplots(1, len(ROBOTS), figsize=(15, 5))

    for i, robot_name in enumerate(ROBOTS):
        result = next(r for r in results
                     if r["map"] == map_name
                     and r["robot"] == robot_name
                     and r["algorithm"] == algorithm)

        if result["path_found"]:
            result["pathfinder"].visualize(result["path"], ax=axes[i])
            axes[i].set_title(f"{robot_name}\nLength: {result['path_length']}m")
        else:
            result["cspace"].visualize(ax=axes[i], title=f"{robot_name}\nNo path")

    plt.suptitle(f"Robot Comparison (Map: {map_name}, Algorithm: {algorithm})", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/robot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_map_overview():
    fig, axes = plt.subplots(1, len(MAPS), figsize=(15, 5))

    for i, map_name in enumerate(MAPS):
        warehouse = load_warehouse(map_name)
        warehouse.visualize(ax=axes[i], title=map_name)

        # Mark start/goal
        start, goal = POSITIONS[map_name]
        axes[i].plot(start[0], start[1], 'go', markersize=10, label='Start')
        axes[i].plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("figures/maps_overview.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_cspace_comparison():
    map_name = "easy"
    warehouse = load_warehouse(map_name)

    fig, axes = plt.subplots(1, len(ROBOTS) + 1, figsize=(16, 4))

    warehouse.visualize(ax=axes[0], title="Warehouse (original)")

    for i, (robot_name, robot) in enumerate(ROBOTS.items()):
        cspace = CSpace(warehouse, robot, resolution=RESOLUTION)
        cspace.visualize(ax=axes[i+1], title=f"C-Space ({robot_name})")

    plt.tight_layout()
    plt.savefig("figures/cspace_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    save_map_overview()
    save_cspace_comparison()
    results = run_all()
    save_results_table(results)
    save_comparison_figures(results)
    save_algorithm_comparison(results)
    save_robot_comparison(results)
