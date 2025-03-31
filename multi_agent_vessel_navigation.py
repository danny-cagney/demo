import heapq
import numpy as np
import time as timer
import random
from typing import List, Tuple, Dict, Set, Optional


# -----------------------------
# Node class for A* pathfinding
# -----------------------------
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None, time=0):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.time = time  # timestep for this node

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return (self.x, self.y, self.time) == (other.x, other.y, other.time)

    def __hash__(self):
        return hash((self.x, self.y, self.time))


# -----------------------------
# Utility Functions
# -----------------------------
def get_location(path, time):
    """Return the location of an agent at a given timestep."""
    if time < len(path):
        return path[time]
    else:
        # Agent stays at the goal location
        return path[-1]


def get_sum_of_cost(paths):
    """Calculate the sum of costs for all agents."""
    return sum(len(path) - 1 for path in paths)


def reconstruct_path(node):
    """Reconstruct the path from the goal node to the start node."""
    path = []
    current = node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]  # Reverse to get start-to-goal path

def build_constraint_table(constraints, agent_id):
    """Preprocess constraints for fast lookup."""
    table = dict()
    for c in constraints:
        if c['agent'] != agent_id:
            continue
        loc_key = tuple(map(tuple, c['loc']))  # ensures [(x, y)] or [(x1, y1), (x2, y2)] is normalized
        key = (loc_key, c['timestep'])
        table[key] = True
    return table

# -----------------------------
# Constraint-Aware A* Algorithm (using original maritime-specific optimizations)
# -----------------------------
def a_star_with_constraints(grid, start, goal, draft, safety_margin, constraints, agent_id,
                            max_timestep, heuristic_weight=1.0, depth_weight=100.0, lookahead=4):
    from heapq import heappush, heappop
    import numpy as np

    class Node:
        def __init__(self, x, y, g=0, time=0):
            self.x = x
            self.y = y
            self.g = g
            self.h = 0
            self.f = 0
            self.time = time
            self.parent = None

        def __lt__(self, other):
            return self.f < other.f

    rows, cols = len(grid), len(grid[0])
    required_depth = draft + safety_margin
    max_depth = max(max(row) for row in grid)
    open_list = []
    closed_set = set()
    g_score = {}

    start_node = Node(*start)
    start_node.h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    start_node.f = heuristic_weight * start_node.h
    heappush(open_list, start_node)

    constraint_table = build_constraint_table(constraints, agent_id)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-way movement

    while open_list:
        current = heappop(open_list)
        state = (current.x, current.y, current.time)
        if state in closed_set:
            continue
        closed_set.add(state)

        if (current.x, current.y) == goal:
            # Reconstruct path
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            depth = grid[nx][ny]
            if depth < required_depth:
                continue

            next_time = current.time + 1
            if next_time > max_timestep:
                continue

            # Normalized constraint check
            vertex_key = (((nx, ny),), next_time)
            edge_key_fwd = (((current.x, current.y), (nx, ny)), next_time)
            edge_key_rev = (((nx, ny), (current.x, current.y)), next_time)

            if vertex_key in constraint_table:
                continue  # Vertex constraint

            if edge_key_fwd in constraint_table or edge_key_rev in constraint_table:
                continue  # Edge constraint (bidirectional)

            # Lookahead depth evaluation
            lookahead_depths = []
            for i in range(1, lookahead + 1):
                look_x = min(nx + i * dx, rows - 1) if dx > 0 else max(nx + i * dx, 0) if dx < 0 else nx
                look_y = min(ny + i * dy, cols - 1) if dy > 0 else max(ny + i * dy, 0) if dy < 0 else ny
                if grid[look_x][look_y] >= required_depth:
                    lookahead_depths.append(grid[look_x][look_y])
                else:
                    break

            if not lookahead_depths:
                continue

            avg_lookahead_depth = sum(lookahead_depths) / len(lookahead_depths)
            depth_bonus = depth_weight * np.log1p(avg_lookahead_depth)
            shallow_penalty = 1000 * np.exp(-depth / max_depth) if max_depth > 0 else 0
            new_g = current.g + 1 - depth_bonus + shallow_penalty

            next_state = (nx, ny, next_time)
            if next_state not in closed_set and (next_state not in g_score or new_g < g_score[next_state]):
                g_score[next_state] = new_g
                neighbor = Node(nx, ny, g=new_g, time=next_time)
                neighbor.h = abs(nx - goal[0]) + abs(ny - goal[1])
                neighbor.f = neighbor.g + heuristic_weight * neighbor.h
                neighbor.parent = current
                heappush(open_list, neighbor)

    return None  # No valid path found

def is_nearby(pos1, pos2, radius):
        """Check if pos2 is within a radius of pos1."""
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 <= radius ** 2

def get_buffer_zone(center, radius, grid_shape):
        """Return list of (x, y) cells within `radius` of center, clipped to grid."""
        x0, y0 = center
        rows, cols = grid_shape
        buffer = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx ** 2 + dy ** 2 <= radius ** 2:
                    x, y = x0 + dx, y0 + dy
                    if 0 <= x < rows and 0 <= y < cols:
                        buffer.append((x, y))
        return buffer

# -----------------------------
# Prioritized Planning Solver
# -----------------------------
class PrioritizedPlanningSolver:
    def __init__(self, depth_grid, starts, goals, vessel_drafts, safety_margins,
                 max_timestep=100, heuristic_weight=1.0, depth_weight=100.0, lookahead=4):
        """
        Initialize a prioritized planning solver for multi-vessel navigation.

        Args:
            depth_grid: 2D grid of depth values
            starts: List of starting positions for each vessel
            goals: List of goal positions for each vessel
            vessel_drafts: List of drafts for each vessel
            safety_margins: List of safety margins for each vessel
            max_timestep: Maximum allowed timestep
            heuristic_weight: Weight for heuristic
            depth_weight: Weight for depth preference
            lookahead: Number of steps to look ahead
        """
        self.depth_grid = depth_grid
        self.starts = starts
        self.goals = goals
        self.vessel_drafts = vessel_drafts
        self.safety_margins = safety_margins
        self.num_of_agents = len(goals)
        self.max_timestep = max_timestep
        self.heuristic_weight = heuristic_weight
        self.depth_weight = depth_weight
        self.lookahead = lookahead
        self.CPU_time = 0

        # Sanity checks
        assert len(starts) == len(goals) == len(vessel_drafts) == len(safety_margins), \
            "Number of starts, goals, vessel drafts, and safety margins must match"

    def find_solution(self):
        """Find paths for all agents using prioritized planning with time-aware moving buffer zones."""
        import time
        import csv

        start_time = time.time()
        result = []
        constraints = []
        agent_metrics = []

        def get_buffer_zone(center, radius, grid_shape):
            x0, y0 = center
            rows, cols = grid_shape
            buffer = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx ** 2 + dy ** 2 <= radius ** 2:
                        x, y = x0 + dx, y0 + dy
                        if 0 <= x < rows and 0 <= y < cols:
                            buffer.append((x, y))
            return buffer

        for i in range(self.num_of_agents):
            print(f"\n🧭 Planning for agent {i} (Vessel {i + 1})...")
            agent_start = time.time()

            path = a_star_with_constraints(
                self.depth_grid,
                self.starts[i],
                self.goals[i],
                self.vessel_drafts[i],
                self.safety_margins[i],
                constraints,
                i,
                self.max_timestep,
                self.heuristic_weight,
                self.depth_weight,
                self.lookahead
            )

            agent_time = time.time() - agent_start
            print(f"⏱️ Agent {i}: Planning time = {agent_time:.2f}s | Constraint count = {len(constraints)}")

            if path is None:
                print(f"❌ No solution found for agent {i}")
                return None

            print(f"✅ Path found for agent {i} with {len(path)} steps")
            result.append(path)

            grid_shape = (len(self.depth_grid), len(self.depth_grid[0]))
            buffer_radius = 3

            for t, loc in enumerate(path):
                buffer_cells = get_buffer_zone(loc, radius=buffer_radius, grid_shape=grid_shape)
                for j in range(i + 1, self.num_of_agents):
                    for buffered_loc in buffer_cells:
                        constraints.append({'agent': j, 'loc': [buffered_loc], 'timestep': t})

                    # Edge constraints
                    if t > 0:
                        prev_loc = path[t - 1]
                        constraints.append({'agent': j, 'loc': [prev_loc, loc], 'timestep': t})
                        constraints.append({'agent': j, 'loc': [loc, prev_loc], 'timestep': t})

            # Add buffered goal constraints for remaining time
            goal_time = len(path) - 1
            goal_buffer = get_buffer_zone(self.goals[i], radius=buffer_radius, grid_shape=grid_shape)
            for j in range(i + 1, self.num_of_agents):
                for t in range(goal_time, self.max_timestep):
                    for buffered_loc in goal_buffer:
                        constraints.append({'agent': j, 'loc': [buffered_loc], 'timestep': t})

            # Save performance metrics
            agent_metrics.append({
                'agent': i,
                'path_length': len(path),
                'planning_time_sec': round(agent_time, 4),
                'constraints_before': len(constraints)
            })

        self.CPU_time = time.time() - start_time
        print("\n🎉 Found solution!")
        print("🧠 Total CPU time (s): {:.2f}".format(self.CPU_time))
        print("🧮 Sum of costs: {}".format(get_sum_of_cost(result)))

        for idx, path in enumerate(result):
            print(f"  Agent {idx}: Path length = {len(path)}")

        # Save CSV metrics
        metrics_file = "agent_planning_metrics.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=agent_metrics[0].keys())
            writer.writeheader()
            writer.writerows(agent_metrics)
        print(f"\n📊 Saved agent metrics to {metrics_file}")

        return result

    def check_collision(self, paths):
        """
        Check if there are any collisions between agents in the given paths.

        Args:
            paths: List of paths for each agent

        Returns:
            True if collision detected, False otherwise
        """
        if not paths:
            return False

        # Determine a simulation horizon
        time_horizon = max(self.max_timestep, max(len(path) for path in paths))

        for t in range(time_horizon):
            occupied = {}
            for agent, path in enumerate(paths):
                pos = get_location(path, t)
                if pos in occupied:
                    print(
                        f"Collision detected at time {t} between agent {agent} and agent {occupied[pos]} at location {pos}.")
                    return True
                occupied[pos] = agent

        return False

    def change_priority_ordering_random(self):
        """Randomly shuffle the priority ordering of the agents."""
        indices = list(range(self.num_of_agents))
        random.shuffle(indices)

        self.starts = [self.starts[i] for i in indices]
        self.goals = [self.goals[i] for i in indices]
        self.vessel_drafts = [self.vessel_drafts[i] for i in indices]
        self.safety_margins = [self.safety_margins[i] for i in indices]


# -----------------------------
# Compute Effective Grid
# -----------------------------
def compute_effective_grid(chart_grid, tide_height):
    """
    Adjust depth grid based on tide height.

    Args:
        chart_grid: 2D grid of depth values at chart datum
        tide_height: Current tide height in same units as chart_grid

    Returns:
        2D grid with adjusted depth values
    """
    return [[(depth + tide_height) if depth > 0 else 0 for depth in row] for row in chart_grid]


# -----------------------------
# Example Usage
# -----------------------------
def run_multi_vessel_example(depth_grid_path, tide_height=0):
    """
    Run an example of multi-vessel path planning.

    Args:
        depth_grid_path: Path to depth grid file (.npy)
        tide_height: Current tide height
    """
    import numpy as np

    # Load depth grid
    depth_grid = np.load(depth_grid_path)

    # Apply tide adjustment
    effective_grid = compute_effective_grid(depth_grid.tolist(), tide_height)

    # Define vessel parameters
    starts = [(10, 5), (50, 10), (30, 80)]
    goals = [(80, 90), (70, 70), (5, 60)]
    vessel_drafts = [5.0, 3.5, 4.0]  # in meters
    safety_margins = [1.0, 1.5, 1.0]  # in meters

    # Initialize and run the solver
    solver = PrioritizedPlanningSolver(
        effective_grid,
        starts,
        goals,
        vessel_drafts,
        safety_margins,
        max_timestep=10000,
        heuristic_weight=1.0,
        depth_weight=110.0,
        lookahead=4
    )

    paths = solver.find_solution()

    # Check for collisions
    if paths and solver.check_collision(paths):
        print("WARNING: Solution has collisions, trying with random priority ordering...")
        solver.change_priority_ordering_random()
        paths = solver.find_solution()

    return paths