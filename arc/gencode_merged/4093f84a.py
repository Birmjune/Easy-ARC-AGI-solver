from common import *

import numpy as np
from typing import *

# concepts:
# attraction, magnetism, color change

# description:
# In the input you will see a grey rectangle and colored pixels scattered around it.
# To make the output, move each colored pixel toward the grey rectangle until it touches, then turn its color to gray. If multiple colored pixels collide, they stack.

def main(input_grid, task = []):
    # Plan:
    # 1. Detect the objects; separate the gray rectangle from the other pixels
    # 2. Move each colored pixel toward the gray rectangle until it touches
    # 3. Change its color once it touches
    def find_background_color(grid):
        """
        그리드의 테두리 픽셀 중에서 가장 많이 등장하는 색상을 반환합니다.
        grid: 2D 리스트 또는 numpy 배열
        """
        grid = np.asarray(grid)
        h, w = grid.shape
        top = grid[0, :]
        bottom = grid[h - 1, :]
        left = grid[:, 0]
        right = grid[:, w - 1]
        edges = np.concatenate([top, bottom, left, right])
        counts = np.bincount(edges.flatten())
        return np.argmax(counts)
    
    def find_unique_element(arr):
        """
        arr: 1D array-like (list or numpy array) of integers (e.g. [np.int64(2), np.int64(6), ...])
        returns: 원소 중에서 유일하게 한 번만 등장하는 값
        """
        arr = np.asarray(arr)
        vals, counts = np.unique(arr, return_counts=True)
        # counts == 1인 값들 중 첫 번째(유일한 값) 반환
        return vals[counts == 1][0]
    background_color = find_background_color(input_grid)

    objects = find_connected_components(input_grid, connectivity=4, background=background_color, monochromatic=True)

    colors = []
    for obj in objects:
        colors.extend(object_colors(obj, background=background_color))

    object_color = find_unique_element(colors)


    grey_objects = [ obj for obj in objects if object_color in object_colors(obj, background=background_color) ]
    other_objects = [ obj for obj in objects if object_color not in object_colors(obj, background=background_color) ]

    assert len(grey_objects) == 1, "There should be exactly one grey object"
    
    grey_object = grey_objects[0]

    # Make the output grid: Start with the gray object, then add the colored pixels one-by-one
    output_grid = np.full_like(input_grid, background_color)
    blit_object(output_grid, grey_object, background=background_color)


    # Move the colored objects and change their color once they hit grey
    for colored_object in other_objects:
        # First calculate what direction we have to move in order to contact the grey object
        # Consider all displacements, starting with the smallest translations first
        possible_displacements = [ (i*dx, i*dy) for i in range(0, 30) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] ]

        # Only keep the displacements that cause a contact between the colored object and the grey object
        valid_displacements = [ displacement for displacement in possible_displacements
                                if contact(object1=translate(colored_object, *displacement), object2=grey_object) ]
        assert valid_displacements, "There should be at least one valid displacement"

        # Pick the smallest valid displacement
        displacement = min(valid_displacements, key=lambda displacement: sum(abs(x) for x in displacement))

        # Extract the direction from the displacement
        direction = np.sign(displacement, dtype=int)

        # Now move the colored object in that direction until there is a collision with something else
        if not all( delta == 0 for delta in direction ):
            while not collision(object1=translate(colored_object, *direction), object2=output_grid):
                colored_object = translate(colored_object, *direction)
        
        # Finally change the color of the colored object to grey anne draw it onto the outlet
        colored_object[colored_object != background_color] = object_color
        blit_object(output_grid, colored_object, background=background_color)
    

    return output_grid

def generate_input():
    # Make a grid with a grey horizontal rectangle stretching all the way through the middle, and some scattered points around it
    # Then randomly rotate to get a variety of orientations

    width, height = np.random.randint(10, 25), np.random.randint(10, 25)
    grid = np.full((width, height), background_color)

    rectangle_y1 = np.random.randint(0, height//2)
    rectangle_y2 = np.random.randint(height//2, height)
    grid[:, rectangle_y1:rectangle_y2] = object_color

    # scatter some colored pixels around the grey rectangle
    for _ in range(np.random.randint(5, 10)):
        random_color = random.choice([color for color in Color.NOT_BLACK if color != object_color])
        pixel_sprite = np.full((1,1), random_color)
        x, y = random_free_location_for_sprite(grid, pixel_sprite, background=background_color)
        blit_sprite(grid, pixel_sprite, x, y, background=background_color)
    
    # random rotation
    grid = np.rot90(grid, np.random.randint(0, 4))

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
