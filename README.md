# Multi-Agent Maritime Vessel Navigation

This project provides a depth-aware, collision-avoiding pathfinding system for multiple maritime vessels. It combines a specialized A* algorithm that accounts for depth requirements with a prioritized planning approach for coordinating multiple vessels.

## Features

- **Depth-Aware Navigation**: Vessels avoid shallow water based on their draft and safety margins
- **Multi-Agent Coordination**: Collision-free paths for multiple vessels using prioritized planning
- **Tide Compensation**: Adjusts depth charts based on tide heights
- **Visualization Tools**: Both static and animated visualization of vessel paths
- **Customizable Parameters**: Configure vessel drafts, safety margins, and pathfinding parameters

## Project Structure

- `multi_agent_vessel_navigation.py`: Core implementation of the A* algorithm and prioritized planning solver
- `path_visualization.py`: Visualization tools for displaying depth grids and vessel paths
- `multi_vessel_main.py`: Command-line interface for running the system
- `example_usage.py`: Example usage of the system with a randomly generated depth grid
- `test_multi_agent_navigation.py`: Unit tests for the system

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/danny-cagney/demo.git
   cd demo
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install numpy matplotlib
   ```

## Usage

### Basic Usage

Run the example script to see the system in action:

```python
python example_usage.py
```

This will generate a random depth grid and plan paths for three vessels.

### Using Your Own Depth Grid

```python
from multi_agent_vessel_navigation import PrioritizedPlanningSolver, compute_effective_grid
from path_visualization import visualize_paths

# Load your depth grid (numpy array)
depth_grid = np.load('your_depth_grid.npy')

# Apply tide adjustment if needed
tide_height = 1.5
effective_grid = compute_effective_grid(depth_grid.tolist(), tide_height)

# Define vessel parameters
starts = [(10, 5), (50, 10), (30, 80)]
goals = [(80, 90), (70, 70), (5, 60)]
vessel_drafts = [5.0, 3.5, 4.0]  # in meters
safety_margins = [1.0, 1.5, 1.0]  # in meters

# Initialize solver
solver = PrioritizedPlanningSolver(
    effective_grid,
    starts,
    goals,
    vessel_drafts,
    safety_margins,
    max_timestep=200,
    heuristic_weight=1.0,
    depth_weight=100.0,
    lookahead=4
)

# Find solution
paths = solver.find_solution()

# Check for collisions
if solver.check_collision(paths):
    print("Warning: Solution has collisions")
    # Optionally try again with different priority ordering
    solver.change_priority_ordering_random()
    paths = solver.find_solution()

# Visualize paths
visualize_paths(depth_grid, paths, vessel_drafts, safety_margins)
```

### Command-Line Interface

You can also use the command-line interface:

```bash
python multi_vessel_main.py --depth-grid your_depth_grid.npy --tide-height 1.5 --save-animation
```

Command-line options:
- `--depth-grid`: Path to the depth grid .npy file (required)
- `--tide-height`: Current tide height in meters (default: 0.0)
- `--max-timestep`: Maximum allowed timestep (default: 200)
- `--save-animation`: Save animation of the vessel paths
- `--save-image`: Save static image of the vessel paths

## How It Works

### Depth-Aware A* Algorithm

The system uses a modified A* algorithm that considers water depth in addition to path length. Key features:

- Considers vessel draft and safety margins when determining navigable areas
- Employs lookahead evaluation to prefer paths that maintain safe depth ahead
- Rewards deeper water with depth bonuses and penalizes shallow water

### Prioritized Planning for Multi-Agent Coordination

To handle multiple vessels without collisions:

1. Vessels are assigned priorities (initially based on input order)
2. Highest-priority vessel plans its path first without constraints
3. Each subsequent vessel plans around the paths of higher-priority vessels
4. If conflicts arise, the system can retry with a different priority ordering

### Constraints

The system uses two types of constraints:
- **Vertex constraints**: Prevent a vessel from occupying a specific location at a specific time
- **Edge constraints**: Prevent a vessel from traveling between two locations at a specific time

## Customization

### Pathfinding Parameters

- `heuristic_weight`: Weight for the goal-directed component (default: 1.0)
- `depth_weight`: Weight for preferring deeper water (default: 100.0)
- `lookahead`: Number of steps to look ahead when evaluating paths (default: 4)
- `max_timestep`: Maximum allowed timestep for planning (default: 100)

### Vessel Parameters

For each vessel, you can specify:
- Starting position
- Goal position
- Draft (minimum required depth)
- Safety margin (additional buffer)

## Testing

Run the unit tests to verify the system is working correctly:

```bash
python test_multi_agent_navigation.py
```

## Example Output

A successful run will produce:
- Collision-free paths for all vessels
- Visualization of the depth grid and vessel paths
- Path statistics (length, planning time, etc.)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
