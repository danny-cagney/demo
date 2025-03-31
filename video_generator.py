import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation
import argparse
import json
import os


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


def create_enhanced_animation(depth_grid, paths, vessel_drafts, safety_margins, filename="vessel_animation.mp4",
                              fps=15, dpi=150, trail_length=20, simulation_speed=1.0):
    """
    Create an enhanced animation of vessel movements along their paths.

    Args:
        depth_grid: 2D numpy array of depth values
        paths: List of paths for each vessel
        vessel_drafts: List of vessel drafts
        safety_margins: List of safety margins
        filename: Output filename (MP4)
        fps: Frames per second
        dpi: Resolution of the output video
        trail_length: How many previous positions to show in the vessel's trail
        simulation_speed: Speed multiplier for the simulation
    """
    if not paths:
        print("No paths to visualize")
        return

    # Find max path length
    max_path_length = max(len(path) for path in paths)
    frames_needed = max_path_length

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(right=0.85)  # Make room for legend and info panel

    # Create a custom colormap for depths - blue palette
    cmap = plt.cm.Blues.copy()
    # Make land areas white instead of red
    cmap.set_under('white')

    # Display depth grid
    max_depth = np.max(depth_grid)
    im = ax.imshow(depth_grid, cmap=cmap, vmin=0.01, vmax=max_depth)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Depth (meters)')

    # Choose distinctive colors for vessels
    vessel_colors = ['red', 'magenta', 'yellow', 'lime', 'cyan', 'orange', 'purple', 'white']
    vessel_markers = []
    vessel_trails = []
    vessel_arrows = []
    vessel_positions = []

    # Initialize vessel visualization elements
    for i, path in enumerate(paths):
        color = vessel_colors[i % len(vessel_colors)]

        # Initial vessel position
        vessel = Circle((path[0][1], path[0][0]), radius=3, color=color, fill=True,
                        zorder=10, label=f'Vessel {i + 1} (Draft: {vessel_drafts[i]}m)')
        ax.add_patch(vessel)
        vessel_markers.append(vessel)

        # Empty trail for now
        trail, = ax.plot([], [], color=color, alpha=0.6, linewidth=2, zorder=5)
        vessel_trails.append(trail)

        # Direction arrow (initially hidden)
        arrow = FancyArrowPatch((0, 0), (0, 0), color=color, linewidth=2,
                                arrowstyle='->', mutation_scale=15, zorder=9, alpha=0)
        ax.add_patch(arrow)
        vessel_arrows.append(arrow)

        # Store all positions for this vessel
        vessel_positions.append([p for p in path])

        # Plot start and goal points
        start = path[0]
        goal = path[-1]
        ax.plot(start[1], start[0], 'o', color=color, markersize=8)
        ax.plot(goal[1], goal[0], 's', color=color, markersize=8)

    # Create info box for simulation details
    info_box = ax.text(0.03, 0.97, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Create a simple legend
    ax.legend(handles=vessel_markers, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Set title and labels
    ax.set_title('Multi-Vessel Navigation Simulation')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    def update(frame):
        # Adjust frame based on simulation speed
        actual_frame = min(int(frame * simulation_speed), frames_needed - 1)

        # Update vessels and trails
        for i, (vessel, trail, arrow) in enumerate(zip(vessel_markers, vessel_trails, vessel_arrows)):
            positions = vessel_positions[i]

            # Current position
            if actual_frame < len(positions):
                current_pos = positions[actual_frame]
                vessel.center = (current_pos[1], current_pos[0])
                vessel.set_alpha(1.0)  # Make vessel fully visible

                # Update trail
                start_idx = max(0, actual_frame - trail_length)
                trail_x = [pos[1] for pos in positions[start_idx:actual_frame + 1]]
                trail_y = [pos[0] for pos in positions[start_idx:actual_frame + 1]]
                trail.set_data(trail_x, trail_y)

                # Update direction arrow
                if actual_frame < len(positions) - 1:
                    next_pos = positions[actual_frame + 1]
                    arrow.set_positions((current_pos[1], current_pos[0]),
                                        (next_pos[1], next_pos[0]))
                    arrow.set_alpha(0.8)
                else:
                    arrow.set_alpha(0)  # Hide arrow at end of path
            else:
                # Vessel has reached its goal
                final_pos = positions[-1]
                vessel.center = (final_pos[1], final_pos[0])

                # Complete trail
                trail_x = [pos[1] for pos in positions[-trail_length:]]
                trail_y = [pos[0] for pos in positions[-trail_length:]]
                trail.set_data(trail_x, trail_y)

                # Hide direction arrow
                arrow.set_alpha(0)

        # Update info box
        progress = min(100, 100 * actual_frame / (frames_needed - 1))
        vessels_moving = sum(1 for i, positions in enumerate(vessel_positions)
                             if actual_frame < len(positions))
        vessels_completed = len(paths) - vessels_moving

        info_text = f"Simulation Progress: {progress:.1f}%\n"
        info_text += f"Timestep: {actual_frame}\n"
        info_text += f"Vessels in motion: {vessels_moving}\n"
        info_text += f"Vessels completed: {vessels_completed}"
        info_box.set_text(info_text)

        return vessel_markers + vessel_trails + vessel_arrows + [info_box]

    # Calculate total number of frames needed
    total_frames = int(frames_needed / simulation_speed) + 30  # Add some extra frames at the end

    # Create animation
    print(f"Creating animation with {total_frames} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / fps, blit=True
    )

    # Save animation
    print(f"Saving animation to {filename}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
    anim.save(filename, writer=writer, dpi=dpi)
    print(f"Animation saved to {filename}")

    # Close figure to free memory
    plt.close(fig)
    print("Animation completed.")


def main():
    """Main function for creating enhanced vessel path animations."""
    parser = argparse.ArgumentParser(description="Create Enhanced Vessel Path Animations")
    parser.add_argument("--grid", type=str, required=True, help="Path to depth grid .npy file")
    parser.add_argument("--positions", type=str, required=True, help="JSON file with vessel positions")
    parser.add_argument("--paths", type=str, help="JSON file with computed paths (optional)")
    parser.add_argument("--output", type=str, default="vessel_animation.mp4", help="Output MP4 filename")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier (default: 1.0)")
    parser.add_argument("--trail", type=int, default=20, help="Trail length (default: 20)")
    parser.add_argument("--dpi", type=int, default=150, help="Video resolution DPI (default: 150)")
    args = parser.parse_args()

    # Load depth grid
    if not os.path.exists(args.grid):
        print(f"Error: Depth grid file '{args.grid}' not found.")
        return

    print(f"Loading depth grid from {args.grid}...")
    depth_grid = np.load(args.grid)

    # Load positions
    positions = load_positions_from_file(args.positions)
    if not positions or 'starts' not in positions or 'goals' not in positions:
        print("Error: Invalid positions file")
        return

    # Convert positions from lists to tuples
    starts = convert_to_tuples(positions.get('starts', []))
    goals = convert_to_tuples(positions.get('goals', []))
    vessel_drafts = positions.get('vessel_drafts', [3.0])
    safety_margins = positions.get('safety_margins', [1.0])

    # Load paths if provided, otherwise compute them
    paths = None
    if args.paths and os.path.exists(args.paths):
        with open(args.paths, 'r') as f:
            paths_data = json.load(f)
            if 'paths' in paths_data:
                paths = [convert_to_tuples(path) for path in paths_data['paths']]
                print(f"Loaded {len(paths)} paths from {args.paths}")

    if paths is None:
        # We need to compute paths
        from multi_agent_vessel_navigation import PrioritizedPlanningSolver

        print("Computing paths using multi-agent solver...")

        # Ensure we have matching parameters
        num_vessels = min(len(starts), len(goals), len(vessel_drafts), len(safety_margins))
        solver = PrioritizedPlanningSolver(
            depth_grid.tolist(),
            starts[:num_vessels],
            goals[:num_vessels],
            vessel_drafts[:num_vessels],
            safety_margins[:num_vessels],
            max_timestep=300,
            lookahead=4
        )

        paths = solver.find_solution()

        if paths is None:
            print("Failed to find valid paths. Try using --paths to provide pre-computed paths.")
            return

        # Save paths for future use
        paths_data = {'paths': [list(map(list, path)) for path in paths]}
        output_path_file = 'computed_paths.json'
        with open(output_path_file, 'w') as f:
            json.dump(paths_data, f, indent=2)
        print(f"Saved computed paths to {output_path_file}")

    # Create the animation
    create_enhanced_animation(
        depth_grid,
        paths,
        vessel_drafts[:len(paths)],
        safety_margins[:len(paths)],
        filename=args.output,
        fps=args.fps,
        dpi=args.dpi,
        trail_length=args.trail,
        simulation_speed=args.speed
    )


if __name__ == "__main__":
    main()