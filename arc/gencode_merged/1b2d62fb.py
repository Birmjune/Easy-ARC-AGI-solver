import numpy as np
from typing import *
from common import *

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two maroon bitmasks separated by a blue vertical bar
# To make the output, color teal the pixels that are not set in either bitmasks (logical NOR)

def main(input_grid: np.ndarray, task = []) -> np.ndarray:
    input_grid = np.asarray(input_grid)
    middle_y = input_grid.shape[1] // 2
    line_color = input_grid[0, middle_y]

    rotated = False
    if (len(np.unique(input_grid[:, middle_y])) != 1):
      rotated = True
      input_grid = np.rot90(input_grid, k=1)
      middle_y = input_grid.shape[1] // 2

    left_mask = input_grid[:, :middle_y]
    right_mask = input_grid[:, middle_y+1:]

    # vote for bg_color
    bg_colors = []
    for i in range (len(task["train"])):
      train_input = np.asarray(task["train"][0]["input"])
      train_output = np.asarray(task["train"][0]["output"])
      bg_colors.append(np.intersect1d(train_input, train_output)[0])
    def most_frequent(arr):
      vals, counts = np.unique(arr, return_counts=True)
      return vals[counts.argmax()]
    bg_color = most_frequent(bg_colors)

    # find fill_color
    train_output1 = np.asarray(task["train"][0]["output"])
    train_output2 = np.asarray(task["train"][0]["output"])
    overlap_colors = np.intersect1d(train_output1, train_output2)

    for overlap_color in overlap_colors:
       if (overlap_color != bg_color):
          fill_color = overlap_color
          break

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask == bg_color) & (right_mask == bg_color)] = fill_color
    
    if (rotated):
       return np.rot90(output_grid, k = -1)
    return output_grid


def generate_input() -> np.ndarray:
    # create a pair of equally sized maroon bitmasks
    width, height = np.random.randint(2, 10), np.random.randint(2, 10)

    grid1 = np.zeros((width, height), dtype=int)
    grid2 = np.zeros((width, height), dtype=int)

    for x in range(width):
        for y in range(height):
            grid1[x, y] = np.random.choice([Color.MAROON, Color.BLACK])
            grid2[x, y] = np.random.choice([Color.MAROON, Color.BLACK])
    
    # create a blue vertical bar
    bar = np.zeros((1, height), dtype=int)
    bar[0, :] = Color.BLUE

    grid = np.concatenate((grid1, bar, grid2), axis=0)

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
