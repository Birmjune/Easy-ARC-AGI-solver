from common import *

import numpy as np
from typing import *

# concepts:
# alignment, objects

# description:
# In the input you will see a red, blue, and yellow shape. Each are the same shape (but different color). They occur left to right in the input grid on a black background, but at different vertical heights.
# The output is the same as the input, but with the vertical heights of the red and yellow shapes adjusted to match the height of the blue shape.

def main(input_grid, task = []):
    input_grid = np.asarray(input_grid)
    # find the blue shape, red shape, and yellow shape
    bg_color = find_background_color(input_grid)
    output_grid = np.full_like(input_grid, bg_color)

    # find main color
    def find_preserved_colors(inputs, outputs, bg_color):
        """
        inputs, outputs: 같은 순서로 대응되는 2D ndarray 리스트. (예: [inp1, inp2, inp3], [out1, out2, out3])
        bg_color: 비교에서 제외할 색상(int)
        반환: 모든 input-output 쌍에서 위치가 동일하게 유지되는 색들의 리스트
        """
        # 첫 번째 input에서 bg_color를 제외한 후보 색들을 뽑아낸 뒤,
        # 나머지 pairs에서도 동일한 색이 있는지 확인
        first_colors = set(np.unique(inputs[0])) - {bg_color}
        preserved = []
        
        for c in first_colors:
            ok = True
            for inp, out in zip(inputs, outputs):
                # c의 위치가 input과 output에서 동일한지 확인
                mask_in  = (inp  == c)
                mask_out = (out == c)
                if not np.array_equal(mask_in, mask_out):
                    ok = False
                    break
            if ok:
                preserved.append(int(c))
        
        return preserved[0]
    train_inputs = [task["train"][i]["input"] for i in range(len(task["train"]))]

    train_outputs = [task["train"][i]["output"] for i in range(len(task["train"]))]
    main_color = find_preserved_colors(train_inputs, train_outputs, bg_color)
    main_coords = np.where(input_grid == main_color)

    color_list = [i for i in range(10) if i != bg_color]

    for color in color_list:
        color_coords = np.where(input_grid == color)
        rows, cols = color_coords

        if rows.size == 0:
            continue
        rows = main_coords[0]
        output_grid[rows, cols] = color

    # blue_coords = np.where(input_grid == Color.BLUE)
    # red_coords = np.where(input_grid == Color.RED)
    # yellow_coords = np.where(input_grid == Color.YELLOW)

    # # set the vertical height of the red and yellow shape to match
    # red_coords = (red_coords[0], blue_coords[1])
    # yellow_coords = (yellow_coords[0], blue_coords[1])

    # # make output grid with the colored shapes at their new locations
    # output_grid = np.full_like(input_grid, Color.BLACK)
    # output_grid[blue_coords] = Color.BLUE
    # output_grid[red_coords] = Color.RED
    # output_grid[yellow_coords] = Color.YELLOW

    return output_grid


def generate_input():

    # All three shapes are the same shape, but different colors, so we generate one sprite and color it three ways
    # We put each sprite in a different grid, and concatenate the grids to make the input grid

    # make a random sprite of size (1-4)x(1-4)
    w = np.random.randint(1, 5)
    h = np.random.randint(1, 5)
    sprite = random_sprite(w, h)

    # Figure out the height of the output grid
    # This has to be the same across all three colors, because we concatenate them along the x axis
    grid_height = np.random.randint(h+1, 16)

    # for each color,
    # put a colored form of the shape in a random spot in a new grid
    subgrids = []
    for color in [Color.BLUE, Color.RED, Color.YELLOW]:
        # make a grid to put the shape in
        # the grid should be wide enough to fit the shape, which has width w
        grid_width = np.random.randint(w, 30//3)
        subgrid = np.full((grid_width, grid_height), Color.BLACK, dtype=int)

        # make the shape that color
        colored_sprite = np.copy(sprite)
        colored_sprite[sprite != Color.BLACK] = color

        # put the shape in a random spot in its grid
        x, y = random_free_location_for_sprite(subgrid, colored_sprite)
        blit_sprite(subgrid, colored_sprite, x, y)
        subgrids.append(subgrid)

    # now concatenate the subgrids along the x axis to make the input grid
    grid = np.concatenate(subgrids, axis=0)
    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
