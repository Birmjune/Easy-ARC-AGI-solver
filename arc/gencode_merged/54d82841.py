from common import *

import numpy as np
from typing import *

# concepts:
# gravity, falling

# description:
# In the input you will see various monochromatic objects
# To make the output, make each object drop a single yellow pixel below it, centered with the middle of the object

def main(input_grid, task = []):
    input_grid = np.asarray(input_grid)
    # Plan:
    # 1. Detect the objects
    # 2. Drop yellow pixels which land in the final row of the grid, centered with the middle of the object
    bg_color = find_background_color(input_grid)

    train_input = task["train"][0]["input"]
    train_output = task["train"][0]["output"]

    def find_unique(a, b):
        """
        a, b: 두 개의 ndarray. 둘 중 하나에만 존재하는 원소가 유일하게 하나라고 가정.
        반환: 그 원소
        """
        # 1) 1차원으로 펼치고 고유값만 뽑아서 set 비교
        sa = set(np.unique(a))
        sb = set(np.unique(b))
        diff = sa ^ sb  # 대칭 차집합: a와 b 중 하나에만 있는 값들의 집합
        return diff.pop()  # 유일한 값 하나를 꺼내서 반환
    new_color = find_unique(train_input, train_output)

    objects = find_connected_components(input_grid, connectivity=4, background=bg_color, monochromatic=True)


    output_grid = input_grid.copy()

    for obj in objects:
        obj_color = find_unique(obj, np.array([[bg_color]]))
        x, y = object_position(obj, background=bg_color, anchor='center')
        if isinstance(x, int):
          if output_grid[x, int(y+1)] == bg_color:    
            bottom_y = output_grid.shape[1] - 1
            output_grid[x, bottom_y] = new_color
          else:
            output_grid[x, 0] = new_color
        if isinstance(y, int):
            if output_grid[int(x+1), y] == bg_color:
                bottom_x = output_grid.shape[0] - 1
                output_grid[bottom_x, y] = new_color
            else:
                top_x = 0
                output_grid[top_x, y] = new_color
    
    return output_grid


def generate_input():
    width, height = np.random.randint(10, 30), np.random.randint(10, 30)
    grid = np.full((width, height), Color.BLACK)
    
    n_objects = np.random.randint(1, 5)
    for _ in range(n_objects):
        # Make a random sprite with odd width, so that there will be a unique horizontally center pixel 
        widths = [3, 5, 7]
        heights = [1,2,3,4]
        sprite = random_sprite(widths, heights, color_palette=[random.choice(Color.NOT_BLACK)])
        # Find a place for it, but put some padding along the borders so that it's not at the bottom
        x, y = random_free_location_for_sprite(grid, sprite, padding=2, border_size=2)
        # Put it down
        blit_sprite(grid, sprite, x, y)

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
