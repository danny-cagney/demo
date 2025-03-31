import heapq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from utils import generate_depth_grid, read_depth_grid_from_csv, read_depth_grid_from_npy
import time as timer

npy_path = "/Users/danielcagney/Desktop/PythonProject/a_star_depth_test/depth_grid.npy"
depth_grid = read_depth_grid_from_npy(npy_path)


# -----------------------------
# A* Algorithm Implementation
# -----------------------------
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

    # Check if start and goal are navigable
    if start not in navigable_cells or goal not in navigable_cells:
        return None  # Early termination if either start or goal is in unsafe water

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

    # Main search loop
    while open_list:
        current = heapq.heappop(open_list)
        current_pos = (current.x, current.y)

        # Skip if already processed
        if current_pos in closed_set:
            continue

        # Check if goal reached
        if current_pos == goal:
            return reconstruct_path(current)

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

            # step_cost = 1 + depth_weight * (1 - depth / max_depth)

            # cumulative cost to reach the new node
            # new_g = current.g + step_cost

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

    # No path found
    return None


# -------------------------------------------
# Effective Grid Computation
# -------------------------------------------
def compute_effective_grid(chart_grid, tide_height):
    return [[(depth + tide_height) if depth > 0 else 0 for depth in row] for row in chart_grid]


# -------------------------------------------
# Visualization Update Function
# -------------------------------------------
def update_visualization(grid, path, vessel_draft, safety_margin):
    global text_annotations, unsafe_patches, path_plot, start_marker, goal_marker, considered_path_plots, legend, no_path_text

    required_depth = vessel_draft + safety_margin
    grid_np = np.array(grid)
    im.set_data(grid_np)

    # Remove previous considered path visualizations safely
    for plot in considered_path_plots:
        if plot:  # Check if plot exists
            try:
                plot.remove()
            except:
                pass  # Silently continue if removal fails
    considered_path_plots.clear()

    # Remove old path markers safely
    if 'path_plot' in globals() and path_plot is not None:
        try:
            path_plot.remove()
        except:
            pass
        path_plot = None
    if 'start_marker' in globals() and start_marker is not None:
        try:
            start_marker.remove()
        except:
            pass
        start_marker = None
    if 'goal_marker' in globals() and goal_marker is not None:
        try:
            goal_marker.remove()
        except:
            pass
        goal_marker = None
    if 'legend' in globals() and legend is not None:
        try:
            legend.remove()
        except:
            pass
        legend = None

    # Remove no path text if it exists
    if 'no_path_text' in globals() and no_path_text is not None:
        try:
            no_path_text.remove()
        except:
            pass
        no_path_text = None

    # Draw final path if it exists (yellow)
    if path:
        path_x = [pt[1] for pt in path]
        path_y = [pt[0] for pt in path]
        path_plot, = ax.plot(path_x, path_y, marker='o', color='yellow', linewidth=0.5, markersize=1,
                             label='Optimal Path')
        start_marker = ax.scatter(path_x[0], path_y[0], color='green', s=50, label='Start')
        goal_marker = ax.scatter(path_x[-1], path_y[-1], color='red', s=50, label='Goal')
    else:
        # Display "No safe path available" text when no path is found
        no_path_text = ax.text(0.5, 0.5, "No safe path available",
                               transform=ax.transAxes,
                               fontsize=16,
                               color='red',
                               weight='bold',
                               ha='center',
                               va='center',
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        # Still show start and end points
        start_marker = ax.scatter(start[1], start[0], color='green', s=50, label='Start')
        goal_marker = ax.scatter(goal[1], goal[0], color='red', s=50, label='Goal')

    # Set new legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='lower left')

    fig.canvas.draw_idle()  # Redraw the canvas


# -----------------------------
# Main Execution with Dynamic Tide Slider
# -----------------------------
if __name__ == "__main__":
    # Initialize globals to avoid undefined variable errors
    text_annotations = []
    unsafe_patches = []
    path_plot = None
    start_marker = None
    goal_marker = None
    considered_path_plots = []
    legend = None
    no_path_text = None

    # Load data
    cork_grid = depth_grid

    # Default parameter values
    default_vessel_draft = 10
    default_safety_margin = 3
    default_tide = 0
    default_heuristic_weight = 1.0
    default_depth_weight = 100
    default_lookahead = 4

    # Initial values (same as defaults)
    vessel_draft = default_vessel_draft
    safety_margin = default_safety_margin
    start = (48, 58)
    goal = (246, 208)

    # Compute initial path
    print("Computing initial path...")
    effective_grid = compute_effective_grid(cork_grid, default_tide)
    start_time = timer.time()
    path = a_star(effective_grid, start, goal, vessel_draft, safety_margin)
    CPU_time = timer.time() - start_time
    print(f"CPU time (s): {CPU_time:.4f}")

    # Create figure and visualization
    print("Setting up visualization...")
    fig, ax = plt.subplots(figsize=(6, 6))
    grid_np = np.array(effective_grid)
    im = ax.imshow(grid_np, cmap='Blues', origin='upper')

    # Adjust the main plot to make room for side panel
    plt.subplots_adjust(left=0.3, bottom=0.05, right=0.95, top=0.95)

    # Create a side panel for algorithm parameters
    params_ax = plt.axes([0.05, 0.25, 0.2, 0.5])
    params_ax.axis('off')
    params_ax.set_title('Algorithm Parameters', fontsize=12, fontweight='bold')

    # Create sliders for algorithm parameters
    tide_ax = plt.axes([0.1, 0.8, 0.15, 0.03])
    tide_slider = Slider(tide_ax, 'Tide Height (m)', valmin=0, valmax=5, valinit=default_tide, valstep=0.1)

    heuristic_ax = plt.axes([0.1, 0.7, 0.15, 0.03])
    heuristic_slider = Slider(heuristic_ax, 'Heuristic Weight', valmin=0.1, valmax=5.0,
                              valinit=default_heuristic_weight, valstep=0.1)

    depth_ax = plt.axes([0.1, 0.6, 0.15, 0.03])
    depth_slider = Slider(depth_ax, 'Depth Weight', valmin=10, valmax=200, valinit=default_depth_weight, valstep=10)

    lookahead_ax = plt.axes([0.1, 0.5, 0.15, 0.03])
    lookahead_slider = Slider(lookahead_ax, 'Lookahead Steps', valmin=1, valmax=10, valinit=default_lookahead,
                              valstep=1)

    # Create slider for vessel parameters
    vessel_ax = plt.axes([0.1, 0.4, 0.15, 0.03])
    vessel_slider = Slider(vessel_ax, 'Vessel Draft (m)', valmin=1, valmax=15, valinit=default_vessel_draft,
                           valstep=0.5)

    safety_ax = plt.axes([0.1, 0.3, 0.15, 0.03])
    safety_slider = Slider(safety_ax, 'Safety Margin (m)', valmin=0, valmax=10, valinit=default_safety_margin,
                           valstep=0.5)

    # Create a reset button
    reset_ax = plt.axes([0.1, 0.2, 0.15, 0.04])
    reset_button = Button(reset_ax, 'Reset Defaults', color='lightgoldenrodyellow', hovercolor='0.975')

    # Initial visualization update
    update_visualization(effective_grid, path, vessel_draft, safety_margin)


    def reset(event):
        """Reset all sliders to default values"""
        print("Resetting parameters to defaults...")
        tide_slider.set_val(default_tide)
        heuristic_slider.set_val(default_heuristic_weight)
        depth_slider.set_val(default_depth_weight)
        lookahead_slider.set_val(default_lookahead)
        vessel_slider.set_val(default_vessel_draft)
        safety_slider.set_val(default_safety_margin)
        print("Parameters reset complete")


    def update(val):
        """Update visualization when sliders change"""
        # Get current values from all sliders
        tide = tide_slider.val
        heuristic_weight = heuristic_slider.val
        depth_weight = depth_slider.val
        lookahead = int(lookahead_slider.val)
        vessel_draft = vessel_slider.val
        safety_margin = safety_slider.val

        # Update the grid with new tide level
        new_effective_grid = compute_effective_grid(cork_grid, tide)

        # Calculate new path with updated parameters
        start_time = timer.time()
        new_path = a_star(new_effective_grid, start, goal, vessel_draft, safety_margin,
                          heuristic_weight=heuristic_weight,
                          depth_weight=depth_weight,
                          lookahead=lookahead)
        CPU_time = timer.time() - start_time

        # Update visualization
        update_visualization(new_effective_grid, new_path, vessel_draft, safety_margin)

        # Print parameter values and results
        print("\n--- Current Parameters ---")
        print(f"Tide Height: {tide:.1f}m")
        print(f"Heuristic Weight: {heuristic_weight:.1f}")
        print(f"Depth Weight: {depth_weight:.1f}")
        print(f"Lookahead Steps: {lookahead}")
        print(f"Vessel Draft: {vessel_draft:.1f}m")
        print(f"Safety Margin: {safety_margin:.1f}m")

        if new_path:
            print(f"Path found with {len(new_path)} waypoints")
        else:
            print("No safe path available")
        print(f"CPU time (s): {CPU_time:.4f}")


    # Connect all sliders to update function
    tide_slider.on_changed(update)
    heuristic_slider.on_changed(update)
    depth_slider.on_changed(update)
    lookahead_slider.on_changed(update)
    vessel_slider.on_changed(update)
    safety_slider.on_changed(update)

    # Connect reset button to reset function
    reset_button.on_clicked(reset)

    # Keep the window open until user closes it
    print("Visualization ready. Close the window to exit.")
    plt.show()