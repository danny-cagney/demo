import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os

# Function to load positions from a JSON file
def load_positions_from_file(filename):
    """Load vessel positions from a JSON file."""
    if not os.path.exists(filename):
        print(f"Warning: Position file {filename} not found")
        return None

    with open(filename, 'r') as f:
        positions = json.load(f)
    print(f"Loaded positions from {filename}")
    return positions

# Convert positions from lists to tuples
def convert_to_tuples(positions_list):
    """Convert positions from lists to tuples for use as dictionary keys."""
    if positions_list is None:
        return None
    return [tuple(pos) for pos in positions_list]

# Function to check if a path is possible
def check_path_feasibility(depth_grid, start, goal, vessel_draft, safety_margin):
    required_depth = vessel_draft + safety_margin
    rows, cols = depth_grid.shape

    # Check if start and goal are within bounds
    if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
        goal[0] < 0 or goal[0] >= rows or goal[1] < 0 or goal[1] >= cols):
        print(f"Start {start} or goal {goal} is out of bounds. Grid shape: {depth_grid.shape}")
        return False

    # Check if start and goal are navigable
    start_depth = depth_grid[start[0], start[1]]
    goal_depth = depth_grid[goal[0], goal[1]]

    print(f"Start position: {start}")
    print(f"Depth at start: {start_depth}")
    print(f"Goal position: {goal}")
    print(f"Depth at goal: {goal_depth}")
    print(f"Required depth: {required_depth}")

    if start_depth < required_depth:
        print(f"Start position {start} is not navigable. Depth: {start_depth}, Required: {required_depth}")
        return False

    if goal_depth < required_depth:
        print(f"Goal position {goal} is not navigable. Depth: {goal_depth}, Required: {required_depth}")
        return False

    # Count the number of navigable cells
    navigable_cells = 0
    for r in range(rows):
        for c in range(cols):
            if depth_grid[r, c] >= required_depth:
                navigable_cells += 1

    print(f"Total navigable cells: {navigable_cells} out of {rows*cols} ({navigable_cells/(rows*cols)*100:.2f}%)")

    # Simple BFS to check if a path is possible
    def bfs_path_exists():
        from collections import deque

        visited = set()
        queue = deque([start])
        visited.add(start)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        while queue:
            current = queue.popleft()
            if current == goal:
                return True

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if (0 <= nx < rows and 0 <= ny < cols and
                    depth_grid[nx, ny] >= required_depth and
                    next_pos not in visited):
                    visited.add(next_pos)
                    queue.append(next_pos)

        print(f"BFS explored {len(visited)} navigable cells but found no path.")
        return False

    path_exists = bfs_path_exists()
    if not path_exists:
        print("No path exists between start and goal using basic BFS algorithm.")
    else:
        print("A path exists between start and goal according to basic BFS algorithm.")

    return path_exists

# Function to visualize the depth grid and positions
def visualize_depth_grid(depth_grid, start=None, goal=None, required_depth=None):
    plt.figure(figsize=(12, 10))

    # Create a custom colormap for depths
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen', 'blue', 'darkblue']
    cmap = LinearSegmentedColormap.from_list('depth_cmap', colors)

    # Display depth grid
    max_depth = np.max(depth_grid)
    im = plt.imshow(depth_grid, cmap=cmap, vmin=0, vmax=max_depth)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Depth (meters)')

    # Mark navigable cells if required_depth is provided
    if required_depth is not None:
        navigable_mask = depth_grid >= required_depth
        plt.contour(navigable_mask, levels=[0.5], colors='white', linestyles='dashed', alpha=0.7)

        # Calculate percentage of navigable cells
        navigable_count = np.sum(navigable_mask)
        total_cells = depth_grid.size
        percentage = navigable_count / total_cells * 100
        plt.title(f'Depth Grid (Navigable: {percentage:.2f}% at depth ≥{required_depth}m)')
    else:
        plt.title('Depth Grid')

    # Mark start and goal
    if start is not None:
        plt.plot(start[1], start[0], 'go', markersize=12, label='Start')

    if goal is not None:
        plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')

    if start is not None or goal is not None:
        plt.legend()

    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('depth_grid_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the depth grid
    if not os.path.exists('depth_grid.npy'):
        print("Error: depth_grid.npy not found")
        return

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

    # Check each vessel
    for i in range(min(len(starts), len(goals), len(vessel_drafts), len(safety_margins))):
        print(f"\n--- Checking vessel {i+1} ---")
        start = starts[i]
        goal = goals[i]
        vessel_draft = vessel_drafts[i]
        safety_margin = safety_margins[i]
        required_depth = vessel_draft + safety_margin

        print(f"Vessel draft: {vessel_draft}m, Safety margin: {safety_margin}m")

        # Visualize the grid with this vessel's requirements
        visualize_depth_grid(depth_grid, start, goal, required_depth)

        # Check if path is possible
        is_feasible = check_path_feasibility(depth_grid, start, goal, vessel_draft, safety_margin)

        if is_feasible:
            print(f"Path should be feasible for vessel {i+1}")
        else:
            print(f"Path is not feasible for vessel {i+1}")

if __name__ == '__main__':
    main()