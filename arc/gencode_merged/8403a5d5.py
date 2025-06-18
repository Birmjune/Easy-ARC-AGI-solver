from common import *

import numpy as np
from typing import *

# concepts:
# pattern generation

# description:
# In the input you will see a grid with a single pixel on the bottom of the grid.
# To make the output, you should draw a geometric pattern starting outward from the pixel:
# step 1: draw vertical bars starting from the pixel and going to the right with a horizontal period of 2.
# step 2: put a grey pixel in between the vertical bars alternating between the top / bottom.

def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    # Output grid is the same size as the input grid
    output_grid = input_grid.copy()
    bg = find_background_color(input_grid)
    # Detect the pixel on the bottom of the grid
    pixel_x, pixel_y = np.argwhere(input_grid != bg)[0]

    # Get the color of the pattern pixel by observation
    pattern_pixel_color = Color.GRAY
    pixel_color = input_grid[pixel_x, pixel_y]
    # STEP 1: Draw vertical bar from bottom to top starting from the pixel and going to the right, horizontal period of 2
    horizontal_period = 2
    for y in range(pixel_y, output_grid.shape[1], horizontal_period):
        draw_line(output_grid, x=pixel_x, y=y, direction=(-1, 0), color=pixel_color)
   
    # STEP 2: put a grey pixel in between the vertical bars alternating between the top / bottom.
    cur_x = -1 if pixel_x == 0 else 0
    for y in range(pixel_y + 1, output_grid.shape[1], horizontal_period):
        output_grid[cur_x, y] = pattern_pixel_color
        # alternate between top and bottom
        cur_x = 0 if cur_x == -1 else -1

    return output_grid

def generate_input():
    # Generate the background grid
    n, m = np.random.randint(10, 20, size=2)
    grid = np.zeros((n, m), dtype=int)

    # Randomly choose the color of the line
    pattern_pixel_color = Color.GRAY
    color = np.random.choice([color for color in Color.NOT_BLACK if color != pattern_pixel_color])

    # Randomly place the pixel on the bottom of the grid
    x = np.random.randint(0, n)
    grid[x, -1] = color

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
