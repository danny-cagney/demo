import argparse
import numpy as np
import os
import time
import random
import json

from multi_agent_vessel_navigation import (
    PrioritizedPlanningSolver,
    compute_effective_grid
)

# Import updated visualization functions with bottom-left legend
from path_visualization import (
    create_enhanced_animation,
    visualize_static_paths,
    visualize_path_blue
)


def generate_random_navigable_positions(depth_grid, vessel_draft, safety_margin, num_positions=1):
    """Generate random navigable positions on the depth grid."""
    required_depth = vessel_draft + safety_margin
    rows, cols = depth_grid.shape

    # Find all navigable cells
    navigable_cells = []
    for r in range(rows):
        for c in range(cols):
            if depth_grid[r, c] >= required_depth:
                navigable_cells.append((r, c))

    if not navigable_cells:
        print(f"No navigable cells found for vessel draft {vessel_draft}m with safety margin {safety_margin}m")
        return []

    # Select random positions
    if len(navigable_cells) <= num_positions:
        return navigable_cells

    return random.sample(navigable_cells, num_positions)


def save_positions_to_file(positions, filename):
    """Save vessel positions to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(positions, f, indent=2)
    print(f"Saved positions to {filename}")


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


def main():
    """Main function with simplified arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Vessel Navigation System")
    parser.add_argument("--grid", type=str, required=True, help="Path to depth grid .npy file")
    parser.add_argument("--positions", type=str, help="JSON file with vessel positions")
    parser.add_argument("--save", type=str, help="Save generated positions to this JSON file")
    parser.add_argument("--vessels", type=int, default=3, help="Number of vessels (default: 3)")
    parser.add_argument("--animate", action="store_true", help="Create animation of vessel paths")
    parser.add_argument("--tide", type=float, default=0.0, help="Tide height in meters (default: 0.0)")
    parser.add_argument("--fps", type=int, default=15, help="Animation frames per second (default: 15)")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed multiplier (default: 1.0)")
    parser.add_argument("--trail", type=int, default=20, help="Trail length in animation (default: 20)")
    args = parser.parse_args()

    # Load depth grid
    if not os.path.exists(args.grid):
        print(f"Error: Depth grid file '{args.grid}' not found.")
        return

    print(f"Loading depth grid from {args.grid}...")
    depth_grid = np.load(args.grid)

    # Apply tide adjustment if specified
    if args.tide > 0:
        print(f"Applying tide height of {args.tide}m...")
        depth_grid = np.array(compute_effective_grid(depth_grid.tolist(), args.tide))

    # Define vessel parameters
    num_vessels = args.vessels

    # Default vessel drafts and safety margins
    vessel_drafts = [3.0, 3.5, 4.0, 4.5, 5.0,][:num_vessels] * 3 # Use first N drafts
    safety_margins = [1.0] * num_vessels

    # Handle positions (starts and goals)
    starts = []
    goals = []

    # If positions file is provided, load it
    if args.positions:
        positions = load_positions_from_file(args.positions)
        if positions and 'starts' in positions and 'goals' in positions:
            # Convert lists to tuples for use as dictionary keys
            starts = convert_to_tuples(positions['starts'])
            goals = convert_to_tuples(positions['goals'])

            # Make sure we adjust the number of vessels
            num_vessels = min(num_vessels, len(starts), len(goals))
            starts = starts[:num_vessels]
            goals = goals[:num_vessels]
            vessel_drafts = vessel_drafts[:num_vessels]
            safety_margins = safety_margins[:num_vessels]

            # If vessel_drafts and safety_margins are in the positions file, use those
            if 'vessel_drafts' in positions and len(positions['vessel_drafts']) >= num_vessels:
                vessel_drafts = positions['vessel_drafts'][:num_vessels]
            if 'safety_margins' in positions and len(positions['safety_margins']) >= num_vessels:
                safety_margins = positions['safety_margins'][:num_vessels]

    # If we don't have positions, generate random ones
    if not starts or not goals:
        print("\nGenerating random navigable positions...")
        all_positions = []

        for i in range(num_vessels):
            # Generate 3 random positions for each vessel
            draft = vessel_drafts[i]
            margin = safety_margins[i]
            random_positions = generate_random_navigable_positions(depth_grid, draft, margin, 3)

            if len(random_positions) < 2:
                print(f"Warning: Not enough navigable positions for vessel {i + 1}")
                if i == 0:
                    # If we can't place the first vessel, abort
                    print("Error: No navigable positions found. Try adjusting vessel drafts.")
                    return
                # Skip this vessel
                continue

            # Use the first position as start and the last as goal
            starts.append(random_positions[0])
            goals.append(random_positions[-1])
            all_positions.append({
                'vessel': i + 1,
                'draft': draft,
                'margin': margin,
                'possible_positions': random_positions,
                'selected_start': random_positions[0],
                'selected_goal': random_positions[-1]
            })

        # Update actual number of vessels
        num_vessels = len(starts)
        vessel_drafts = vessel_drafts[:num_vessels]
        safety_margins = safety_margins[:num_vessels]

        print("\nGenerated positions for each vessel:")
        for i, pos in enumerate(all_positions):
            print(f"  Vessel {i + 1} (Draft: {pos['draft']}m, Safety: {pos['margin']}m):")
            print(f"    Possible positions: {pos['possible_positions']}")
            print(f"    Selected start: {pos['selected_start']}")
            print(f"    Selected goal: {pos['selected_goal']}")

        # Save positions to file if requested
        if args.save:
            save_positions = {
                'starts': starts,
                'goals': goals,
                'vessel_drafts': vessel_drafts,
                'safety_margins': safety_margins,
                'all_possible_positions': [p['possible_positions'] for p in all_positions]
            }
            save_positions_to_file(save_positions, args.save)

    # If we still don't have valid positions, abort
    if not starts or not goals or len(starts) != len(goals) or len(starts) == 0:
        print("Error: Failed to generate or load valid vessel positions.")
        return

    print(f"\nPlanning paths for {num_vessels} vessels:")
    for i in range(num_vessels):
        print(
            f"  Vessel {i + 1}: Start={starts[i]}, Goal={goals[i]}, Draft={vessel_drafts[i]}m, Safety={safety_margins[i]}m")

    # Initialize solver
    solver = PrioritizedPlanningSolver(
        depth_grid.tolist(),
        starts,
        goals,
        vessel_drafts,
        safety_margins,
        max_timestep=300,  # Increased max timestep for complex grids
        lookahead=4  # Default lookahead
    )

    # Find solution
    print("\nFinding solution...")
    start_time = time.time()
    paths = solver.find_solution()
    planning_time = time.time() - start_time

    if paths is None:
        print("Failed to find a solution. Trying with random priority ordering...")

        # Try again with a different priority ordering
        solver.change_priority_ordering_random()
        print("Using new random priority ordering:")
        for i in range(solver.num_of_agents):
            print(
                f"  Vessel {i + 1}: Start={solver.starts[i]}, Goal={solver.goals[i]}, Draft={solver.vessel_drafts[i]}m")

        start_time = time.time()
        paths = solver.find_solution()
        planning_time = time.time() - start_time

        if paths is None:
            print("Still failed to find a solution. Try different positions or fewer vessels.")
            return

    print(f"\nPlanning completed in {planning_time:.2f} seconds")

    # Check for collisions
    if solver.check_collision(paths):
        print("WARNING: Solution has collisions!")
    else:
        print("Verified: Solution is collision-free.")

    # Save paths for future use
    paths_data = {'paths': [list(map(list, path)) for path in paths]}
    output_path_file = 'computed_paths.json'
    with open(output_path_file, 'w') as f:
        json.dump(paths_data, f, indent=2)
    print(f"Saved computed paths to {output_path_file}")

    # Visualize the complete solution statically
    print("\nCreating static visualization...")
    visualize_static_paths(
        depth_grid,
        paths,
        vessel_drafts,
        safety_margins,
        save_image=True,
        filename="multi_vessel_paths.png"
    )

    # Create enhanced animation if requested
    if args.animate:
        print("\nCreating enhanced animation...")
        animation_filename = "multi_vessel_animation.mp4"
        create_enhanced_animation(
            depth_grid,
            paths,
            vessel_drafts,
            safety_margins,
            filename=animation_filename,
            fps=args.fps,
            simulation_speed=args.speed,
            trail_length=args.trail
        )
        print(f"Animation saved to {animation_filename}")


if __name__ == "__main__":
    main()