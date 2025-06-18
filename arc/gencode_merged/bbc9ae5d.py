from common import *

import numpy as np
from typing import *

# concepts:
# counting, incrementing

# description:
# In the input, you will see a row with partially filled with pixels from the one color from left to right.
# To make the output: 
# 1. Take the input row
# 2. Copy it below the original row with one more colored pixel added to the sequence if there is space
# 3. Repeat until there are half as many rows as there are columns
def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    # get the color of the row
    color = input_grid[0, 0]
    nonzero_indices = np.where(input_grid[0] != color)[0]
    first = nonzero_indices[0]
    first_color = input_grid[0, first]
    # copy the row from the input grid
    row = np.copy(input_grid)

    # make the output grid
    output_grid = np.copy(input_grid)
    # repeat the row on the output grid until there are half as many rows as there are columns
    for i in range(len(input_grid[0])//2 - 1):
        # find the rightmost color pixel in the row and add one more pixel of the same color to the right if there is space
        row2 = np.full_like(row, color)
        for j in range(len(input_grid[0])-1):
            row2[0, j+1] = row[0, j]
        row = row2
        # add the row to the output grid
        output_grid = np.concatenate((output_grid, row2), axis=0)

    return output_grid

def generate_input():
    # decide the length of the row, and make it even
    length = np.random.randint(3, 8) * 2

    # decide the color to partially fill the row with
    color = random.choice(list(Color.NOT_BLACK))

    # make a row with the color partially filled from left to right
    row = np.zeros((length,1), dtype=int)
    num_colored_pixels = np.random.randint(1, length//2 + 2)
    row[:num_colored_pixels,:] = color

    # make this row the entire grid
    grid = row

    return grid



# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)