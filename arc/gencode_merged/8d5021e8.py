from common import *

import numpy as np
from typing import *

# concepts:
# flip

# description:
# In the input you will see a monochromatic sprite.
# To make the output, 
# 1. flip the grid horizontally with y-axis on the left side of the grid, making the canvas twice larger.
# 2. flip it down with x-axis on the bottom side of the grid.
# 3. concatenate the flipped grid in step 2 to the top and bottom of the grid in step 1.
# In total the output grid is twice as wide and three times as tall as the input.

def main(input_grid, task = []):
    # Create the output grid twice as wide and three times as tall
    input_grid = np.asarray(input_grid)
    n, m = input_grid.shape
    output_grid = np.zeros((n * 3, m * 2), dtype=int)
 
    # Step 1: Flip the grid horizontally with y-axis on the left side of the grid, concate it to the left.
    # Place it in the middle of the output grid
    flip_grid = np.fliplr(input_grid)
    blit_sprite(output_grid, sprite=flip_grid, x=n, y=0)
    blit_sprite(output_grid, sprite=input_grid, x=n, y=m)

    # Step 2: Flip it down with x-axis on the bottom side of the grid, concate it to the bottom and top.
    original_object = output_grid[n : 2 * n, :]
    filp_down_object = np.flipud(original_object)
    blit_sprite(output_grid, sprite=filp_down_object, x=0, y=0)
    blit_sprite(output_grid, sprite=filp_down_object, x=2 * n, y=0)
    
    return output_grid

def generate_input():
    # Generate grid of size n x m
    n, m = np.random.randint(2, 6), np.random.randint(2, 6)
    grid = np.zeros((n, m), dtype=int)

    # Randomly choose 1 color
    color = np.random.choice(list(Color.NOT_BLACK))

    # Randomly choose the density of the color
    density = np.random.randint(2, n * m) / (n * m)

    # Randomly scatter the color in the grid
    randomly_scatter_points(grid, color=color, density=density)

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
