import time
import random
from typing import List

# Build a matrix
def load_graph(file_path: str, n: int = 1000):
    dist = [[0.0] * n for _ in range(n)]

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts[0].isdigit():
                continue
            try:
                i = int(parts[0]) - 1 # Strat from 0 to fit the format of a matrix
                j = int(parts[1]) - 1
                d = float(parts[2])
            except:
                continue
            dist[i][j] = d
            dist[j][i] = d
    return dist

# Calculate the distance of tours
def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    n = len(tour)
    total_dist = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        total_dist += dist[i][j]
    return total_dist

# Apply the closest insertion method
def closest_insertion_with_start(dist: List[List[float]], start: int) -> List[int]:
    n = len(dist)

    # Start with a 2-node cycle
    nearest = min(
        (j for j in range(n) if j != start),
        key=lambda j: dist[start][j]
    )
    tour = [start, nearest]
    unvisited = set(range(n))
    unvisited.remove(start)
    unvisited.remove(nearest)

    # Insert nodes repeatedly
    while unvisited:
        k = min(
            unvisited,
            key=lambda x: min(dist[x][t] for t in tour)
        )

        best_pos = None
        best_up = float("inf")
        m = len(tour)

        for i in range(m):
            a = tour[i]
            b = tour[(i + 1) % m]
            increase = dist[a][k] + dist[k][b] - dist[a][b]
            if increase < best_up:
                best_up = increase
                best_pos = i + 1

        tour.insert(best_pos, k)
        unvisited.remove(k)

    return tour

# Generate the tour string
def tour_cycle(tour: List[int]) -> str:
    if not tour:
        return ""
    tour = [v + 1 for v in tour] + [tour[0] + 1]
    return ", ".join(str(v) for v in tour)

if __name__ == "__main__":
    EDGE_FILE = "TSP_1000_randomDistance.txt"
    TIME_BUDGET = 60.0

    print("Loading distance matrix...")
    dist_matrix = load_graph(EDGE_FILE, n=1000)

    print("\nRunning Closest Insertion...")

    ci_start_time = time.time()
    ci_best_tour: List[int] | None = None
    ci_best_cost = float("inf")
    ci_cycles = 0

    while True:
        if time.time() - ci_start_time >= TIME_BUDGET:
            break

        start_node = random.randrange(1000)
        tour = closest_insertion_with_start(dist_matrix, start_node)
        cost = tour_length(tour, dist_matrix)

        ci_cycles += 1

        if cost < ci_best_cost:
            ci_best_cost = cost
            ci_best_tour = tour

    print(f"\nBest Closest Insertion cost: {ci_best_cost:.2f}")
    print(f"Closest Insertion tours evaluated: {ci_cycles:.0e}")

    if ci_best_tour is not None:
        print("\nBest Closest Insertion tour:")
        print(tour_cycle(ci_best_tour))
    else:
        print("No tour found.")