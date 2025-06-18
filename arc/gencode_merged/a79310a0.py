from common import *

import numpy as np
from typing import *

# concepts:
# translation, color change

# description:
# In the input you will see a grid with a teal object.
# To make the output grid, you should translate the teal object down by 1 pixel and change its color to red.

def main(input_grid: np.ndarray, task = []) -> np.ndarray:
    input_grid = np.asarray(input_grid)
    # Plan:
    # 1. Find the object (it's the only one)
    # 2. Change its color to red
    # 3. Translate it downward by 1 pixel
    bg_color = find_background_color(input_grid)

    train_input = task["train"][0]["input"]
    train_output = task["train"][0]["output"]
    def find_unique_in_A_not_in_B(A, B):
      """
      A, B: 2D numpy ndarray
      A에만 있는 원소가 유일하다고 가정했을 때, 그 원소를 반환
      """
      setA = set(np.unique(A))
      setB = set(np.unique(B))
      return (setA - setB).pop()
    new_color = find_unique_in_A_not_in_B(train_output, train_input)
        
    # Get the single teal object
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=False, background=bg_color)

    # Make a blank output grid
    output_grid = np.full(input_grid.shape, bg_color)

    for teal_object in objects:
        # Change its color to red
        teal_object[teal_object != bg_color] = new_color

        # Translate it downward by 1 pixel
        teal_object = translate(teal_object, x=1, y=0, background=bg_color)

        # Blit the teal object onto the output grid
        output_grid = blit_object(grid=output_grid, obj=teal_object, background=bg_color)

    return output_grid

def generate_input():
    # Generate the background grid with size of n x n.
    grid_len = np.random.randint(4, 8)
    grid = np.zeros((grid_len, grid_len), dtype=int)

    # Randomly generate the teal object and place it on the grid.
    sprite_width, sprite_height = np.random.randint(1, grid_len - 1), np.random.randint(1, grid_len -1)
    sprite = random_sprite(n=sprite_width, m=sprite_height, color_palette=[Color.TEAL], density=0.5)
    x, y = random_free_location_for_sprite(grid=grid, sprite=sprite, border_size=1)
    grid = blit_sprite(x=x, y=y, grid=grid, sprite=sprite, background=bg_color)
    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
