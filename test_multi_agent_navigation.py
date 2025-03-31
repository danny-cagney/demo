import numpy as np
import matplotlib.pyplot as plt
import time

from multi_agent_vessel_navigation import (
    PrioritizedPlanningSolver,
    compute_effective_grid,
    a_star_with_constraints
)


def create_simple_grid():
    """Create a simple test grid with a clear path."""
    grid = np.zeros((20, 20))

    # Add a deep water channel
    grid[5:15, :] = 10.0

    # Add some shallow areas
    grid[8:12, 5:7] = 3.0

    # Add some obstacles
    grid[8:12, 10:12] = 0.0

    return grid


def test_single_vessel():
    """Test pathfinding for a single vessel."""
    grid = create_simple_grid()

    # Define vessel parameters
    start = (10, 2)
    goal = (10, 18)
    vessel_draft = 2.0
    safety_margin = 1.0

    # Try finding a path
    path = a_star_with_constraints(
        grid.tolist(),
        start,
        goal,
        vessel_draft,
        safety_margin,
        [],  # No constraints
        0,  # Agent ID
        max_timestep=100,
        lookahead=2
    )

    if path:
        print("Test successful! Found path:")
        print(path)

        # Visualize the path
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='Blues')
        plt.colorbar(label='Depth')

        y_coords, x_coords = zip(*path)
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)
        plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
        plt.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')

        plt.title(f'Vessel Path (Draft: {vessel_draft}m, Safety Margin: {safety_margin}m)')
        plt.legend()
        plt.savefig('single_vessel_test.png')
        plt.show()

        return True
    else:
        print("Test failed! Couldn't find a path.")
        return False


def test_multi_vessel():
    """Test pathfinding for multiple vessels."""
    grid = create_simple_grid()

    # Define vessel parameters
    starts = [(7, 2), (10, 2), (13, 2)]
    goals = [(7, 18), (10, 18), (13, 18)]
    vessel_drafts = [2.0, 3.0, 2.0]
    safety_margins = [1.0, 1.0, 1.0]

    # Create solver
    solver = PrioritizedPlanningSolver(
        grid.tolist(),
        starts,
        goals,
        vessel_drafts,
        safety_margins,
        max_timestep=100,
        lookahead=2
    )

    # Find solution
    start_time = time.time()
    paths = solver.find_solution()
    planning_time = time.time() - start_time

    if paths:
        print(f"Test successful! Found paths in {planning_time:.2f} seconds:")
        for i, path in enumerate(paths):
            print(f"Vessel {i + 1}: {path}")

        # Visualize the paths
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='Blues')
        plt.colorbar(label='Depth')

        colors = ['r', 'g', 'b']
        for i, path in enumerate(paths):
            y_coords, x_coords = zip(*path)
            plt.plot(x_coords, y_coords, f'{colors[i % 3]}-', linewidth=2,
                     label=f'Vessel {i + 1} (Draft: {vessel_drafts[i]}m)')
            plt.plot(starts[i][1], starts[i][0], f'{colors[i % 3]}o', markersize=8)
            plt.plot(goals[i][1], goals[i][0], f'{colors[i % 3]}s', markersize=8)

        plt.title('Multi-Vessel Navigation Paths')
        plt.legend()
        plt.savefig('multi_vessel_test.png')
        plt.show()

        return True
    else:
        print("Test failed! Couldn't find paths for all vessels.")
        return False


if __name__ == "__main__":
    print("Testing single vessel navigation...")
    single_test_result = test_single_vessel()

    print("\nTesting multi-vessel navigation...")
    multi_test_result = test_multi_vessel()

    if single_test_result and multi_test_result:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Check the output for details.")