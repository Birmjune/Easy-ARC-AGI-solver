from common import *

import numpy as np
from typing import *

# concepts:
# object detection, pattern drawing

# description:
# In the input you will see two blue pixels
# To make the output grid, you should place a 3x3 green cross pattern between the two blue pixels exactly halfway between them.

def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    # Detect the two blue pixels on the grid.
    background_color = find_background_color(input_grid)
    colored_pixels = np.argwhere(input_grid != background_color)
    first_x, first_y = colored_pixels[0]
    second_x, second_y = colored_pixels[1]
    mid_x, mid_y = int((first_x + second_x) / 2), int((first_y + second_y) / 2)

    # Generate the 3x3 green cross pattern between the two blue pixels.
    green_cross_sprite = np.array( [[background_color, Color.GREEN,background_color],
                                [Color.GREEN, Color.GREEN, Color.GREEN],
                                [background_color, Color.GREEN,background_color]])
    output_grid = input_grid.copy()
    green_cross_width, green_cross_height = 3, 3

    # Put the cross centered at the midpoint
    upper_left_x = mid_x - green_cross_width // 2
    upper_left_y = mid_y - green_cross_height // 2
    blit_sprite(output_grid, green_cross_sprite, x=upper_left_x, y=upper_left_y, background=background_color)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    grid_len = np.random.randint(10, 15)
    n, m = grid_len, grid_len
    grid = np.zeros((n, m), dtype=int)

    # Randomly choose if the two blue pixels are horizontal or vertical.
    if_horizontal  = np.random.choice([True, False])

    # Randomly place two blue pixels on the grid and ensure a 3x3 green cross pattern can be placed in the middle of them.
    line_interval = random.choice(range(1, 4))
    line_width = 3 + line_interval * 2

    while(line_width + 2 > grid_len):
        line_interval = random.choice(range(1, 4))
        line_width = 3 + line_interval * 2

    # Ensure there is enough space for the 3 x 3 green cross pattern and two blue pixels.
    if if_horizontal:
        x = np.random.randint(0, n - line_width - 1)
        y = np.random.randint(1, m - 1)
        grid[x, y] = Color.BLUE
        grid[x + line_width + 1, y] = Color.BLUE
    else:
        x = np.random.randint(1, n - 1)
        y = np.random.randint(0, m - line_width - 1)
        grid[x, y] = Color.BLUE
        grid[x, y + line_width + 1] = Color.BLUE

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
