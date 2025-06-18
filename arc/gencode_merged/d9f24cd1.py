from common import *

import numpy as np
from typing import *

# concepts:
# line drawing, obstacle avoidance

# description:
# In the input you will see several red pixels on the bottom row of the grid, and some gray pixels scattered on the grid.
# To make the output grid, you should draw a red line upward from each red pixel, but avoiding the gray pixels.
# To avoid touching the gray pixels, go right to avoid them until you can go up again.

def main(input_grid, task = []):
    input_grid = np.asarray(input_grid)
    # The output grid is the same size as the input grid, and we are going to draw on top of the input, so we copy it
    bg_color = find_background_color(input_grid)
    def find_boundary_colors(arr):
      """
      2D ndarray arr의 경계(첫/마지막 행, 첫/마지막 열)에 등장하는 색을 리스트로 반환
      """
      top    = arr[0, :]
      bottom = arr[-1, :]
      left   = arr[:, 0]
      right  = arr[:, -1]
      edges  = np.concatenate([top, bottom, left, right])
      return np.unique(edges).tolist()
    
    boundary_colors = find_boundary_colors(input_grid)
    for color in boundary_colors:
        if (color != bg_color):
            fill_color = color
            break
    colors = np.unique(input_grid).tolist()
    for color in colors:
        if (color != bg_color and color != fill_color):
            block_color = color
            break

    def find_fill_side(input_grid, fill_color):
      """
      input_grid에서 fill_color가 있는 모든 좌표를 보고,
      그 좌표들이 전부 한쪽(위/아래/왼쪽/오른쪽) 경계 위에만 있다면
      그 쪽 이름을 반환합니다. 그렇지 않으면 빈 리스트를 반환합니다.
      """
      ys, xs = np.argwhere(input_grid == fill_color).T
      if ys.size == 0:
          return []
      
      n_rows, n_cols = input_grid.shape
      sides = []
      
      # 모든 좌표가 y==0이면 "top"
      if np.all(ys == 0):
          sides.append("top")
      # 모든 좌표가 y==n_rows-1이면 "bottom"
      if np.all(ys == n_rows - 1):
          sides.append("bottom")
      # 모든 좌표가 x==0이면 "left"
      if np.all(xs == 0):
          sides.append("left")
      # 모든 좌표가 x==n_cols-1이면 "right"
      if np.all(xs == n_cols - 1):
          sides.append("right")
      
      return sides[0]
    
    rotation = find_fill_side(input_grid=input_grid, fill_color=fill_color)

    if (rotation == "bottom"):
        k = 0
    if (rotation == "left"):
        k = 1
    if (rotation == "top"):
        k = 2
    if (rotation == "right"):
        k = 3

    for i in range(k):
        input_grid = np.rot90(input_grid, 1)
    
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # Get the positions of the red pixels on the bottom row
    for x, y in np.argwhere(input_grid == fill_color):
        # Draw the red line upward, but move to the right to avoid touching gray pixels
        while 0 < y < height and 0 < x < width:
            if output_grid[x -1, y] == block_color:
                # If the red line touch the gray pixel, it should go right then up to avoid the gray pixel.
                output_grid[x, y+1] = fill_color
                y += 1
            else:
                # Otherwise we go up
                output_grid[x-1, y] = fill_color
                x -= 1
    for i in range(k):
        output_grid = np.rot90(output_grid, -1)
    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = 10, 10
    grid = np.zeros((n, m), dtype=int)

    # Generate the red pixels on the bottom row.
    # Get 3 random positions for the red pixels.
    available_postion = range(1, 9)
    red_location = random.sample(available_postion, 3)

    # Draw the red pixels on the bottom row.
    for pos_x in red_location:
        grid[pos_x, -1] = fill_color
    
    # Get the region except the bottom row, left most column and right most column.
    # Randomly scatter the gray pixels on the grid.
    randomly_scatter_points(grid[1:-1, 1:-1], color=Color.GRAY, density=0.1)
    
    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)