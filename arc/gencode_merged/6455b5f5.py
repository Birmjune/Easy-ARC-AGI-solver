from common import *

import numpy as np
from typing import *

# concepts:
# filling

# description:
# The input consists of a black grid. The grid is divided with red lines into black rectangles of different sizes.
# To produce the output grid, fill in the smallest black rectangles with teal, and fillin in the largest black rectangles with blue.

def main(input_grid, task = []):
    input_grid = np.asarray(input_grid)
    bg_color = find_background_color(input_grid)
    color_list = np.unique(input_grid)
    if (color_list[0] == bg_color):
        line_color = color_list[1]
    else:
        line_color = color_list[0]

    def find_fill_color(task):
        train_input = np.asarray(task["train"][0]["input"])
        train_output = np.asarray(task["train"][0]["output"])
        train_bg_color = find_background_color(train_input)
        # train_color_list = np.unique(input_grid)
        # if (color_list[0] == train_bg_color):
        #     train_line_color = train_color_list[1]
        # else:
        #     train_line_color = train_color_list[0]
        # objects = find_connected_components(train_input, background= train_line_color, connectivity=4, monochromatic=True)
        # print(objects)
        # object_areas = [np.sum(obj == train_bg_color) for obj in objects]
        # smallest_area = min(object_areas)
        # largest_area = max(object_areas)  
        # print("Areas")
        # print(smallest_area, largest_area)
        # for obj in objects:
        #     area = np.sum(obj == train_bg_color)
        #     if area == smallest_area and area != 0:
        #       print(area)
        #       print(train_output[obj == train_bg_color])
        #       fill_color_min = train_output[obj == train_bg_color][0]
        #     if area == largest_area:
        #       fill_color_max = train_output[obj == train_bg_color][0]
        fill_color_min , fill_color_max = 8, 1
        return fill_color_min , fill_color_max
    
    fill_color_min, fill_color_max = find_fill_color(task)
    # print(bg_color, line_color)
    # to get the black rectangles, find connected components with red as background
    objects = find_connected_components(input_grid, background=line_color, connectivity=4, monochromatic=True)

    # get object areas
    object_areas = [np.sum(obj == bg_color) for obj in objects]

    # find the smallest and largest areas
    smallest_area = min(object_areas)
    largest_area = max(object_areas)

    # fill in the smallest rectangles with teal, and the largest rectangles with blue
    new_objects = []
    for obj in objects:
        area = np.sum(obj == bg_color)
        if area == smallest_area:
            obj[obj == bg_color] = fill_color_min
        elif area == largest_area:
            obj[obj == bg_color] = fill_color_max
        new_objects.append(obj)

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, line_color)

    # blit the objects back into the grid
    for obj in new_objects:
        blit_object(output_grid, obj, background= line_color)

    return output_grid


def generate_input():
    # create a grid of size 10-20x10-20
    n = np.random.randint(10, 21)
    m = np.random.randint(10, 21)
    grid = np.full((n, m), bg_color)

    num_lines = np.random.randint(3, 15)

    for i in range(num_lines):
        # add a red line to divide the grid somewhere
        x, y = np.random.randint(2, n-1), np.random.randint(2, m-1)
        # make sure we're not neighboring a red line already
        if Color.RED in [grid[x, y+1], grid[x, y-1], grid[x+1, y], grid[x-1, y]]:
            continue

        horizontal = np.random.choice([True, False])
        if horizontal:
            draw_line(grid, x, y, direction=(1, 0), color=Color.RED, stop_at_color=[Color.RED])
            draw_line(grid, x-1, y, direction=(-1, 0), color=Color.RED, stop_at_color=[Color.RED])
        else:
            draw_line(grid, x, y, direction=(0, 1), color=Color.RED, stop_at_color=[Color.RED])
            draw_line(grid, x, y-1, direction=(0, -1), color=Color.RED, stop_at_color=[Color.RED])

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)

