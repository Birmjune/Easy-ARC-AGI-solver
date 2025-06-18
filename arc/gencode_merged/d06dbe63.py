from common import *

import numpy as np
from typing import *

# concepts:
# staircase pattern

# description:
# In the input you will see a single teal pixel.
# To make the output, draw a staircase from the teal pixel to the upper right and lower left with a step size of 2.

def main(test_input, task = []):
    input_grid = np.asarray(test_input)
    # Find the location of the teal pixel
    def most_common_int(arr):
      values, counts = np.unique(arr, return_counts=True)
      return values[np.argmax(counts)]
    background_color = most_common_int(input_grid)
    colored_pixels = np.argwhere(input_grid != background_color)
    object_color = input_grid[colored_pixels[0]]
    # we are going to draw on top of the input
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # staircase is gray NO! it's pink
    staircase_color = Color.GRAY

    # Draw stairs from the teal pixel
    STAIR_LEN = 2
    # First draw stair to the upper right
    
    for teal_x, teal_y in colored_pixels:
        x, y = teal_x, teal_y
        while 0 <= x < width and 0 <= y < height:
            draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(1, 0))
            x += STAIR_LEN
            # go up
            draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(0, -1))
            y -= STAIR_LEN
            # go right
            
        
        # Then draw stair to the lower left
        x, y = teal_x, teal_y
        while 0 <= x < width and 0 <= y < height:
            draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(-1, 0))
            x -= STAIR_LEN
            # go down
            draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(0, 1))
            y += STAIR_LEN
            # go left
    
    for x, y in colored_pixels:  
        # make sure that the teal pixel stays there
        color = input_grid[x,y]
        output_grid[x, y] = color
      
    return output_grid
    

def generate_input():
    # Generate grid
    width, height = np.random.randint(15, 25), np.random.randint(15, 25)
    grid = np.zeros((width, height), dtype=int)

    # Randomly place one teal pixel on the grid
    # Ensure the pixel is not on the border
    x, y = np.random.randint(width // 3, width * 2 // 3), np.random.randint(height // 3, height * 2 // 3)
    grid[x, y] = Color.TEAL

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
