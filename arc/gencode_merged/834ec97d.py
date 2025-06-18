from common import *

import numpy as np
from typing import *

# concepts:
# geometric pattern

# description:
# In the input you will see a grid with a single coloured pixel.
# To make the output, move the colored pixel down one pixel and draw a yellow line from the pixel to the top of the grid.
# Finally repeat the yellow line by repeating it horizontally left/right with a period of 2 pixels.

def main(input_grid, task = []):
    # Plan:
    # 1. Extract the pixel from the input grid
    # 2. Move the pixel one pixel down
    # 3. Draw a yellow line from the pixel to the top of the grid, repeating it horizontally left/right with a period of 2 pixels
    input_grid = np.asarray(input_grid)
    # 1. Extract the pixel
    background_color = find_background_color(input_grid)
    colored_pixels = np.argwhere(input_grid != background_color)
    pixel_x, pixel_y = colored_pixels[0]
    pixel_color = input_grid[pixel_x, pixel_y]
    # 2. Move the pixel one pixel down
    output_grid = input_grid.copy()
    output_grid[pixel_x + 1, pixel_y] = pixel_color
    output_grid[pixel_x, pixel_y] = background_color

    # 3. Draw the vertical line from the pixel to top

    # Draw the line from left to right
    horizontal_period = 2
    for y in range(pixel_y, output_grid.shape[1], horizontal_period):
        draw_line(output_grid, x=pixel_x, y=y, direction=(-1, 0), color=Color.YELLOW)

    # Draw the line from left to right
    for y in range(pixel_y, -1, -horizontal_period):
        draw_line(output_grid, x=pixel_x, y=y, direction=(-1, 0), color=Color.YELLOW)
    return output_grid

def generate_input():
    # Generate the background grid
    width, height = np.random.randint(5, 30, size=2)
    grid = np.zeros((width, height), dtype=int)

    # Randomly choose one color
    color = np.random.choice([color for color in Color.NOT_BLACK if color != Color.YELLOW])

    # Randomly place the pixel on the grid
    x, y = np.random.randint(0, width - 1), np.random.randint(0, height - 1)
    grid[x, y] = color

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
