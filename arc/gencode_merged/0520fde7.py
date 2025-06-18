import numpy as np
from typing import *
from common import *

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two blue bitmasks separated by a grey vertical bar
# To make the output, color teal the red that are set in both bitmasks (logical AND)

def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    w, h = input_grid.shape
    if len(np.unique(input_grid[w//2+1])) == 1:
        x_coords = 1
    else:
        x_coords = 0
    if x_coords == 1:
        # Find the grey vertical bar. Vertical means constant X
        for x_bar in range(input_grid.shape[0]):
            if np.all(input_grid[x_bar, :] == Color.GREY):
                break

        left_mask = input_grid[:x_bar, :]
        right_mask = input_grid[x_bar+1:, :]

        output_grid = np.zeros_like(left_mask)
        output_grid[(left_mask == color_left) & (right_mask == color_right)] = Color.RED
    else:
        # Find the grey vertical bar. Horizontal means constant Y
        for y_bar in range(input_grid.shape[1]):
            if np.all(input_grid[:, y_bar] == Color.GREY):
                break
        left_mask = input_grid[w//2]
        right_mask = input_grid[2, y_bar+1:]
        output_grid = np.zeros_like(left_mask)
        output_grid[(left_mask == color_left) & (right_mask == color_right)] = Color.RED
        
        return output_grid


def generate_input() -> np.ndarray:
    # create a pair of equally sized maroon bitmasks
    width, height = np.random.randint(2, 10), np.random.randint(2, 10)

    grid1 = np.zeros((width, height), dtype=int)
    grid2 = np.zeros((width, height), dtype=int)

    for x in range(width):
        for y in range(height):
            grid1[x, y] = np.random.choice([Color.BLUE, Color.BLACK])
            grid2[x, y] = np.random.choice([Color.BLUE, Color.BLACK])
    
    # create a blue vertical bar
    bar = np.zeros((1, height), dtype=int)
    bar[0, :] = Color.GREY

    grid = np.concatenate((grid1, bar, grid2), axis=0)

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
