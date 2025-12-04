import time
import random
from typing import List, Tuple

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

# Build the quick tour using nearest neighbor(NN) algorithm
def nn_algorithm(dist: List[List[float]], start: int) -> List[int]:
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        # Find the lowest cost one
        next_city = min(unvisited, key=lambda j: dist[current][j])
        unvisited.remove(next_city)
        tour.append(next_city)
        current = next_city
    return tour

# Optimize with 2-opts algorithm
def two_opt(
    tour: List[int],
    dist: List[List[float]],
    time_limit: float,
    start_time: float
) -> Tuple[List[int], float]:
    n = len(tour)
    best_length = tour_length(tour, dist)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if time.time() - start_time >= time_limit:
                    return tour, best_length
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                old_cost = dist[a][b] + dist[c][d]
                new_cost = dist[a][c] + dist[b][d]
                if new_cost + 1e-12 < old_cost:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    best_length += (new_cost - old_cost)
                    improved = True
    return tour, best_length

# start from random nodes to get the optimal solution
def tsp_solver(
    dist: List[List[float]],
    time_budget: float = 60.0,
    seed: int = 0
) -> Tuple[List[int], float, int]:
    random.seed(seed)
    n = len(dist)
    start_time = time.time()
    best_tour = None
    best_length = float("inf")
    cycles_evaluated = 0
    while True:
        duration = time.time() - start_time
        if duration >= time_budget:
            break
        start_node = random.randrange(n)

        # Using NN-algorithm
        tour = nn_algorithm(dist, start_node)
        length = tour_length(tour, dist)
        cycles_evaluated += 1

        # Improve with 2-opt algorithm
        duration = time.time() - start_time
        remaining = time_budget - duration
        if remaining <= 0:
            # Do not meet the limitation
            if length < best_length:
                best_length = length
                best_tour = tour.copy()
            break

        tour, length = two_opt(tour, dist, remaining, time.time())
        cycles_evaluated += 1

        if length < best_length:
            best_length = length
            best_tour = tour.copy()
    return best_tour, best_length, cycles_evaluated

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

    print("\nRunning a random-start Nearest Neighbor ...")

    nn_start_time = time.time()
    nn_best_tour = None
    nn_best_cost = float("inf")
    nn_cycles = 0

    while True:
        if time.time() - nn_start_time >= TIME_BUDGET:
            break

        start_node = random.randrange(1000)
        tour = nn_algorithm(dist_matrix, start_node)
        cost = tour_length(tour, dist_matrix)
        nn_cycles += 1
        if cost < nn_best_cost:
            nn_best_cost = cost
            nn_best_tour = tour

    print(f"Best NN cost: {nn_best_cost:.2f}")
    print(f"NN cycles evaluated: {nn_cycles:.0e}")
    print("Best NN tour:")
    print(tour_cycle(nn_best_tour))

    print("\nSolving TSP with NN+2-opts ...")

    best_tour, best_cost, cycles = tsp_solver(
        dist_matrix,
        time_budget=TIME_BUDGET,
        seed=24
    )

    if best_tour is None:
        print("No tour found (something went wrong).")
    else:
        print(f"Best tour cost: {best_cost:.2f}")
        print(f"Cycles evaluated: {cycles:.0e}")

        cycle_str = tour_cycle(best_tour)
        print("\nBest tour:")
        print(cycle_str)



