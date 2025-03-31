import numpy as np
import matplotlib.pyplot as plt
import json
import heapq
import time as timer
import os


class Node:
    def __init__(self, x, y, g=0, h=0, parent=None, direction=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.direction = direction

    def __lt__(self, other):
        return self.f < other.f


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]


def a_star(grid, start, goal, vessel_draft, safety_margin, heuristic_weight=1, depth_weight=100, lookahead=4):
    """
    Optimized A* pathfinding for maritime navigation with efficient handling of land/zero-depth areas.

    Args:
        grid: 2D array of depth values (0 for land/unmapped)
        start: Tuple (row, col) of start position
        goal: Tuple (row, col) of goal position
        vessel_draft: Minimum depth required for the vessel
        safety_margin: Additional safety buffer for depth
        heuristic_weight: Weight for the goal-directed component
        depth_weight: Weight for preferring deeper water
        lookahead: Number of steps to look ahead when evaluating paths

    Returns:
        List of coordinates forming the path, or None if no path exists
    """
    required_depth = vessel_draft + safety_margin
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # No diagonal movement

    # Pre-compute navigable cells (optimization for grids with many zeros)
    np_grid = np.array(grid)
    max_depth = np.max(np_grid)
    navigable_cells = set()

    for r in range(rows):
        for c in range(cols):
            if np_grid[r, c] >= required_depth:
                navigable_cells.add((r, c))

    print(f"Found {len(navigable_cells)} navigable cells out of {rows * cols} total cells")
    print(f"Navigable percentage: {len(navigable_cells) / (rows * cols) * 100:.2f}%")

    # Check if start and goal are navigable
    start_depth = np_grid[start[0], start[1]] if 0 <= start[0] < rows and 0 <= start[1] < cols else 0
    goal_depth = np_grid[goal[0], goal[1]] if 0 <= goal[0] < rows and 0 <= goal[1] < cols else 0

    print(f"Start position: {start}, depth: {start_depth}m (required: {required_depth}m)")
    print(f"Goal position: {goal}, depth: {goal_depth}m (required: {required_depth}m)")

    if start not in navigable_cells:
        print(f"Start position is not navigable!")
        return None  # Early termination if start is in unsafe water

    if goal not in navigable_cells:
        print(f"Goal position is not navigable!")
        return None  # Early termination if goal is in unsafe water

    # Initialize start node
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    start_node.h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    start_node.f = start_node.g + heuristic_weight * start_node.h

    # Initialize search structures
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()
    g_score = {start: 0}

    # Debug counters
    nodes_expanded = 0
    max_queue_size = 1

    # Main search loop
    print("Starting A* search...")
    while open_list:
        max_queue_size = max(max_queue_size, len(open_list))
        current = heapq.heappop(open_list)
        current_pos = (current.x, current.y)
        nodes_expanded += 1

        # Status update every 10,000 nodes
        if nodes_expanded % 10000 == 0:
            print(f"Expanded {nodes_expanded} nodes so far, current queue size: {len(open_list)}")

        # Skip if already processed
        if current_pos in closed_set:
            continue

        # Check if goal reached
        if current_pos == goal:
            path = reconstruct_path(current)
            print(
                f"Path found! Length: {len(path)}, Nodes expanded: {nodes_expanded}, Max queue size: {max_queue_size}")
            return path

        closed_set.add(current_pos)

        # Process neighbors
        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            neighbor_pos = (nx, ny)

            # Quickly skip invalid or non-navigable cells
            if (nx < 0 or nx >= rows or ny < 0 or ny >= cols or
                    neighbor_pos not in navigable_cells):
                continue

            depth = grid[nx][ny]

            # Lookahead evaluation for path quality
            lookahead_depths = []
            for i in range(1, lookahead + 1):
                look_x = min(nx + i * dx, rows - 1) if dx > 0 else max(nx + i * dx, 0) if dx < 0 else nx
                look_y = min(ny + i * dy, cols - 1) if dy > 0 else max(ny + i * dy, 0) if dy < 0 else ny
                if grid[look_x][look_y] >= required_depth:
                    lookahead_depths.append(grid[look_x][look_y])
                else:
                    break  # Stop lookahead if we hit shallow/land

            # Skip if no valid lookahead path
            if not lookahead_depths:
                continue

            avg_lookahead_depth = sum(lookahead_depths) / len(lookahead_depths)

            # Calculate costs with logarithmic depth bonus and exponential shallow penalty
            depth_bonus = depth_weight * np.log1p(avg_lookahead_depth)
            shallow_penalty = 1000 * np.exp(-depth / max_depth)
            new_g = current.g + 1 - depth_bonus + shallow_penalty

            # Skip if we already have a better path
            if neighbor_pos in g_score and g_score[neighbor_pos] <= new_g:
                continue

            # Create and add neighbor to open list
            g_score[neighbor_pos] = new_g
            neighbor = Node(nx, ny, g=new_g)
            neighbor.h = abs(nx - goal_node.x) + abs(ny - goal_node.y)
            neighbor.f = neighbor.g + heuristic_weight * neighbor.h
            neighbor.parent = current

            heapq.heappush(open_list, neighbor)

    print(f"No path found after expanding {nodes_expanded} nodes, max queue size: {max_queue_size}")
    return None  # No path found


def compute_effective_grid(chart_grid, tide_height):
    return [[(depth + tide_height) if depth > 0 else 0 for depth in row] for row in chart_grid]


def load_positions_from_file(filename):
    """Load vessel positions from a JSON file."""
    if not os.path.exists(filename):
        print(f"Warning: Position file {filename} not found")
        return None

    with open(filename, 'r') as f:
        positions = json.load(f)
    print(f"Loaded positions from {filename}")
    return positions


def convert_to_tuples(positions_list):
    """Convert positions from lists to tuples for use as dictionary keys."""
    if positions_list is None:
        return None
    return [tuple(pos) for pos in positions_list]


def visualize_path(depth_grid, path, start, goal, vessel_draft, safety_margin):
    """Visualize the depth grid and path."""
    required_depth = vessel_draft + safety_margin
    plt.figure(figsize=(12, 10))

    # Create a custom depth-based colormap
    plt.imshow(depth_grid, cmap='Blues', origin='upper')
    plt.colorbar(label='Depth (meters)')

    # Mark navigable areas
    navigable = depth_grid >= required_depth
    plt.contour(navigable, levels=[0.5], colors='white', linestyles='dashed', alpha=0.7)

    # Plot start and goal
    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')

    # Plot path if it exists
    if path:
        path_x = [pt[1] for pt in path]
        path_y = [pt[0] for pt in path]
        plt.plot(path_x, path_y, 'y-', linewidth=2, markersize=4, marker='o', label=f'Path ({len(path)} points)')

    plt.title(f'Depth Grid and Path (Vessel Draft: {vessel_draft}m, Safety: {safety_margin}m)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.legend()
    plt.savefig('path_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to debug the original A* algorithm with the provided data."""
    # Load the depth grid
    if not os.path.exists('depth_grid.npy'):
        print("Error: depth_grid.npy not found")
        return

    print("Loading depth grid...")
    depth_grid = np.load('depth_grid.npy')
    print(f"Loaded depth grid with shape {depth_grid.shape}")

    # Basic statistics about the grid
    print(f"Depth range: {np.min(depth_grid):.2f} to {np.max(depth_grid):.2f} meters")
    print(f"Mean depth: {np.mean(depth_grid):.2f} meters")
    print(f"Percentage of obstacles (zero depth): {np.sum(depth_grid == 0) / depth_grid.size * 100:.2f}%")

    # Load positions
    if not os.path.exists('agents.json'):
        print("Error: agents.json not found")
        return

    positions = load_positions_from_file('agents.json')
    if not positions:
        return

    # Convert positions from lists to tuples
    starts = convert_to_tuples(positions.get('starts', []))
    goals = convert_to_tuples(positions.get('goals', []))
    vessel_drafts = positions.get('vessel_drafts', [3.0])
    safety_margins = positions.get('safety_margins', [1.0])

    # Try with different tide heights to see if that helps
    tide_heights = [0, 1, 2, 3, 4, 5]

    # Check each vessel
    for i in range(min(len(starts), len(goals), len(vessel_drafts), len(safety_margins))):
        print(f"\n======== Testing Vessel {i + 1} ========")
        start = starts[i]
        goal = goals[i]
        vessel_draft = vessel_drafts[i]
        safety_margin = safety_margins[i]

        print(f"Vessel draft: {vessel_draft}m, Safety margin: {safety_margin}m")

        # Try different tide heights
        for tide in tide_heights:
            print(f"\n--- With tide height {tide}m ---")
            effective_grid = compute_effective_grid(depth_grid.tolist(), tide)

            # Try the original A* algorithm
            start_time = timer.time()
            path = a_star(effective_grid, start, goal, vessel_draft, safety_margin)
            end_time = timer.time()

            print(f"A* search took {end_time - start_time:.2f} seconds")

            if path:
                print(f"Success! Path found with tide height {tide}m")
                visualize_path(np.array(effective_grid), path, start, goal, vessel_draft, safety_margin)
                break
            else:
                print(f"No path found with tide height {tide}m")

        if not path:
            print(f"Failed to find a path for vessel {i + 1} at any tide height")
            # Still visualize the grid to see the problem
            visualize_path(np.array(effective_grid), None, start, goal, vessel_draft, safety_margin)


if __name__ == '__main__':
    main()