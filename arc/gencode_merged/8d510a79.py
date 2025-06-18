from common import *

import numpy as np
from typing import *

# concepts:
# magnetism, lines

# description:
# In the input, you will see a horizontal grey line on a black background, with red and blue pixels scattered on either side of the line.
# To make the output, draw vertical lines from each of the blue and red pixels, with lines from the red pixels going toward the grey line and lines from the blue pixels going away from the grey line. 
# These lines should stop when they hit the grey line or the edge of the grid.

def main(input_grid, task=[]):
    input_grid = np.asarray(input_grid)
    colors = color_list(input_grid)
    colors = colors[colors != 1 ]
    colors = colors[colors != 2 ]
    bg = colors[-1]
    line = colors[0]
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the location of the horizontal grey line
    grey_line = np.where(output_grid == line)
    x_coords = np.unique(grey_line[0])
    y_coords = np.unique(grey_line[1])

    # 수직선 vs 수평선 판별
    if len(y_coords) == 1:  # 수직선 (x값은 다양하고 y값이 동일)
        is_vertical = True
        line_pos = y_coords[0]
    elif len(x_coords) == 1:  # 수평선
        is_vertical = False
        line_pos = x_coords[0]
    else:
        raise ValueError("grey_line이 수직선도 수평선도 아님")

    # get the unique y-coordinates of the grey line
    grey_line_y = np.unique(grey_line[1])

    # find the red and blue pixels
    red_pixels = np.where(output_grid == 2)
    blue_pixels = np.where(output_grid == 1)

    if is_vertical:
        # 수직선일 때: 수선의 발을 왼쪽/오른쪽으로 그림
        for i in range(len(red_pixels[0])):
            x, y = red_pixels[0][i], red_pixels[1][i]
            if y < line_pos:
                draw_line(output_grid, x, y, length=None, color=2, direction=(0, 1), stop_at_color=[line])
            else:
                draw_line(output_grid, x, y, length=None, color=2, direction=(0, -1), stop_at_color=[line])

        for i in range(len(blue_pixels[0])):
            x, y = blue_pixels[0][i], blue_pixels[1][i]
            if y < line_pos:
                draw_line(output_grid, x, y, length=None, color=1, direction=(0, -1), stop_at_color=[line])
            else:
                draw_line(output_grid, x, y, length=None, color=1, direction=(0, 1), stop_at_color=[line])

    else:
        # 수평선일 때: 수선의 발을 위쪽/아래쪽으로 그림
        for i in range(len(red_pixels[0])):
            x, y = red_pixels[0][i], red_pixels[1][i]
            if x < line_pos:
                draw_line(output_grid, x, y, length=None, color=2, direction=(1, 0), stop_at_color=[line])
            else:
                draw_line(output_grid, x, y, length=None, color=2, direction=(-1, 0), stop_at_color=[line])

        for i in range(len(blue_pixels[0])):
            x, y = blue_pixels[0][i], blue_pixels[1][i]
            if x < line_pos:
                draw_line(output_grid, x, y, length=None, color=1, direction=(-1, 0), stop_at_color=[line])
            else:
                draw_line(output_grid, x, y, length=None, color=1, direction=(1, 0), stop_at_color=[line])

    return output_grid

def generate_input():
    # make a 10x10 black grid for the background
    n = m = 10
    grid = np.zeros((n,m), dtype=int)

    # make a horizontal grey line on a random row about halfway down the grid
    row = np.random.randint(m//3, 2*(m//3))
    grid[:, row] = Color.GREY

    # scatter a random number of blue and red pixels on either side of the grey line so that no pixel is in the same column as any other pixel on its side of the grey line
    # select columns for the pixels above the grey line
    cols = np.random.choice(np.arange(m), size=np.random.randint(3, 7), replace=False)
    for col in cols:
      # randomly choose whether to make the pixel red or blue
      if np.random.rand() < 0.5:
        grid[col, np.random.randint(row-1)] = 1
      else:
        grid[col, np.random.randint(row-1)] = 2
    # select columns for the pixels below the grey line
    cols = np.random.choice(np.arange(m), size=np.random.randint(3, 7), replace=False)
    for col in cols:
      # randomly choose whether to make the pixel red or blue
      if np.random.rand() < 0.5:
        grid[col, np.random.randint(row+1, m)] = 1
      else:
        grid[col, np.random.randint(row+1, m)] = 2 
    
    return grid




# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)