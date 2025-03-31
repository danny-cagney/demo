# Description: Utility functions for reading and processing depth data from a GeoTIFF file.
import csv
import numpy as np
import rasterio
from skimage.measure import block_reduce

def generate_depth_grid(tif_path, y_min, y_max, x_min, x_max, grid_size=(250, 250), save_path=None):
    with rasterio.open(tif_path) as src:
        # Read only the necessary window
        window = rasterio.windows.Window(x_min, y_min, x_max - x_min, y_max - y_min)
        depth_data = src.read(1, window=window)
        depth_data[depth_data == src.nodata] = np.nan  # Replace no-data values with NaN

    # Handle NaN values: Replace with 0 before averaging
    cropped_depth = np.nan_to_num(depth_data, nan=0)

    # Resize to grid_size using block averaging
    block_size_y = cropped_depth.shape[0] // grid_size[0]
    block_size_x = cropped_depth.shape[1] // grid_size[1]

    # Use skimage block_reduce to average over blocks
    depth_grid = block_reduce(cropped_depth, block_size=(block_size_y, block_size_x), func=np.mean)

    # Normalize: Replace any remaining NaN with 0
    depth_grid = np.nan_to_num(depth_grid, nan=0)

    # Convert negative values to absolute values
    depth_grid = np.abs(depth_grid)

    # Save the processed grid to a file if save_path is provided
    if save_path:
        np.save(save_path, depth_grid)

    # Convert to a Python 2D list
    return depth_grid.tolist()

# Example usage
tif_path = "/Users/danielcagney/Desktop/PythonProject/a_star_depth_test/BY_KRY12_05_CorkHarbour_2m_U29N_LAT_TIFF_Inshore_Ireland/BY_KRY12_05_CorkHarbour_2m_U29N.tif"
y_min, y_max = 3000, 8000
x_min, x_max = 6000, 10000
save_path = "/Users/danielcagney/Desktop/PythonProject/a_star_depth_test/depth_grid.npy"

depth_grid_list = generate_depth_grid(tif_path, y_min, y_max, x_min, x_max, save_path=save_path)
# print(depth_grid_list)

def read_depth_grid_from_csv(csv_path):
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        depth_grid = [[float(value) for value in row] for row in reader]
    return depth_grid

def read_depth_grid_from_npy(npy_path):
    return np.load(npy_path).tolist()