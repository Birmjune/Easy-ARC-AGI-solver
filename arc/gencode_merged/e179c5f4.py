from common import *

import numpy as np
from typing import *

# concepts:
# bouncing

# description:
# In the input you will see a single blue pixel on a black background
# To make the output, shoot the blue pixel diagonally up and to the right, having it reflect and bounce off the walls until it exits at the top of the grid. Finally, change the background color to teal.

def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    bg = find_background_color(input_grid)
    input_grid = input_swap_color(input_grid, [[bg, 0]])
    colors = np.unique(input_grid)
    color = colors[colors != 0][0]
    # Plan:
    # 1. Detect the pixel
    # 2. Shoot each line of the reflection one-by-one, bouncing (changing horizontal direction) when it hits a (horizontal) wall/edge of canvas
    # 3. Change the background color to teal

    # 1. Find the location of the pixel
    blue_pixels = np.argwhere(input_grid == color)
    for blue_pixel in blue_pixels:
        blue_pixel_x, blue_pixel_y = blue_pixel
        # 2. do the bounce which requires keeping track of the direction of the ray we are shooting, as well as the tip of the ray
        # initially we are shooting diagonally up and to the right (dx=1, dy=-1)
        # initially the tip of the ray is the blue pixel, x=blue_pixel_x, y=blue_pixel_y
        if (blue_pixel_x, blue_pixel_y) == (0,0):
            direction = (1, 1)
        elif  (blue_pixel_x, blue_pixel_y) == (input_grid.shape[0]-1, 0):
            direction = (-1, 1)
        elif  (blue_pixel_x, blue_pixel_y) == (0, input_grid.shape[1]-1):
            direction = (1, -1)
        else:
            direction = (-1,-1)
        cnt = 0
        # loop until we fall out of the canvas
        while 0 <= blue_pixel_x < input_grid.shape[0] and 0 <= blue_pixel_y < input_grid.shape[1]:
            cnt += 1
            stop_x, stop_y = draw_line(input_grid, blue_pixel_x, blue_pixel_y, direction=direction, color=color)
            # print(blue_pixel_x, blue_pixel_y, stop_x, stop_y)
            # Terminate if we failed to make progress
            if cnt == 2 or stop_x == blue_pixel_x and stop_y == blue_pixel_y:
                break
            blue_pixel_x, blue_pixel_y = stop_x, stop_y
            if stop_x == 0 or stop_x == input_grid.shape[0]-1:
                direction = (-direction[0], direction[1])
            elif stop_y == 0 or stop_y == input_grid.shape[1]-1:
                direction = (direction[0], -direction[1])
            else:
                # print("!!!")
                break
    
    old_background = Color.BLACK
    new_background = Color.TEAL
    input_grid[input_grid == old_background] = new_background
    
    return input_grid


def generate_input():
    width, height = np.random.randint(2, 15), np.random.randint(10, 30)
    grid = np.full((width, height), Color.BLACK)
    grid[0,-1] = Color.BLUE

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
