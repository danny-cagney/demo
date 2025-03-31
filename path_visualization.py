import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation


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

        # Initial vessel position - REDUCED SIZE from 3 to 1.5
        vessel = Circle((path[0][1], path[0][0]), radius=1.5, color=color, fill=True,
                        zorder=10, label=f'Vessel {i + 1} (Draft: {vessel_drafts[i]}m)')
        ax.add_patch(vessel)
        vessel_markers.append(vessel)

        # Empty trail for now
        trail, = ax.plot([], [], color=color, alpha=0.6, linewidth=1.5, zorder=5)  # Thinner line
        vessel_trails.append(trail)

        # Direction arrow (initially hidden) - smaller arrow
        arrow = FancyArrowPatch((0, 0), (0, 0), color=color, linewidth=1.5,
                                arrowstyle='->', mutation_scale=10, zorder=9, alpha=0)
        ax.add_patch(arrow)
        vessel_arrows.append(arrow)

        # Store all positions for this vessel
        vessel_positions.append([p for p in path])

        # Plot start and goal points - smaller markers
        start = path[0]
        goal = path[-1]
        ax.plot(start[1], start[0], 'o', color=color, markersize=4)  # Reduced from 8
        ax.plot(goal[1], goal[0], 's', color=color, markersize=4)  # Reduced from 8

    # Create legend in the bottom left corner
    legend = ax.legend(handles=vessel_markers, loc='lower left', fontsize=9,
                       bbox_to_anchor=(0.01, 0.02), frameon=True)

    # Create info box for simulation details - place to the right of the legend
    # First, get the width of the legend to calculate the x position
    fig.canvas.draw()  # This is needed to compute the legend size
    legend_width = legend.get_window_extent().transformed(ax.transAxes.inverted()).width

    # Position the info box to the right of the legend with a small gap
    info_box = ax.text(legend_width + 0.05, 0.02, '', transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

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
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
        anim.save(filename, writer=writer, dpi=dpi)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("If FFmpeg is not installed, please install it for video creation.")
        print("Alternatively, using 'pillow' writer for GIF output:")
        try:
            anim.save(filename.replace('.mp4', '.gif'), writer='pillow', dpi=dpi // 2, fps=fps // 2)
            print(f"GIF animation saved instead.")
        except Exception as gif_error:
            print(f"Failed to create GIF as well: {gif_error}")

    # Close figure to free memory
    plt.close(fig)
    print("Animation completed.")


def visualize_static_paths(depth_grid, paths, vessel_drafts, safety_margins, save_image=False,
                           filename="multi_vessel_paths.png"):
    """
    Create a static visualization of all vessel paths on the depth grid.

    Args:
        depth_grid: 2D numpy array of depth values
        paths: List of paths for each vessel
        vessel_drafts: List of vessel drafts
        safety_margins: List of safety margins
        save_image: Whether to save the image to a file
        filename: Name of the file to save the image to
    """
    if not paths:
        print("No paths to visualize")
        return

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

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

    # Plot paths
    vessel_colors = ['red', 'magenta', 'yellow', 'lime', 'cyan']
    line_styles = ['-', '--', '-.', ':']

    for i, path in enumerate(paths):
        color = vessel_colors[i % len(vessel_colors)]
        linestyle = line_styles[i % len(line_styles)]

        # Extract x and y coordinates for plotting
        y_coords, x_coords = zip(*path)

        # Plot the path - thinner lines
        ax.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                linewidth=1.5, label=f'Vessel {i + 1} (Draft: {vessel_drafts[i]}m)')

        # Mark start and goal - smaller markers
        ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=3)
        ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=3)

        # Add timestep labels at regular intervals - with smaller font
        if len(path) > 10:
            interval = len(path) // 5
            for t in range(0, len(path), interval):
                if t > 0 and t < len(path) - 1:  # Don't label start/end
                    ax.text(x_coords[t], y_coords[t], f't={t}',
                            color=color, fontsize=7, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, pad=0.5))

    # Set title and labels
    ax.set_title('Multi-Vessel Navigation Paths')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Place legend in bottom left corner
    ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()

    if save_image:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Image saved to {filename}")

    plt.show()


def visualize_path_blue(depth_grid, path, start, goal, vessel_draft, safety_margin, title="Path Visualization",
                        save_image=False, filename="optimal_path.png"):
    """Visualize a single path with a blue depth color scheme."""
    required_depth = vessel_draft + safety_margin
    plt.figure(figsize=(12, 10))

    # Create a blue colormap with white for land
    cmap = plt.cm.Blues.copy()
    cmap.set_under('white')

    # Display depth grid
    plt.imshow(depth_grid, cmap=cmap, vmin=0.01, vmax=np.max(depth_grid))
    plt.colorbar(label='Depth (meters)')

    # Mark navigable areas
    navigable = depth_grid >= required_depth
    plt.contour(navigable, levels=[0.5], colors='lightblue', linestyles='dashed', alpha=0.5)

    # Plot start and goal - smaller markers
    plt.plot(start[1], start[0], 'go', markersize=3, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=3, label='Goal')

    # Plot path if it exists - thinner line and smaller markers
    if path:
        path_x = [pt[1] for pt in path]
        path_y = [pt[0] for pt in path]
        plt.plot(path_x, path_y, 'y-', linewidth=1, markersize=1, marker='o', label=f'Optimal Path')

    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')

    # Place legend in bottom left corner
    plt.legend(loc='lower left', fontsize=9)

    if save_image:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Image saved to {filename}")

    plt.show()