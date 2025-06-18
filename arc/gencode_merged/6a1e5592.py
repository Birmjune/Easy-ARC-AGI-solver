from common import *

import numpy as np
from typing import *

# concepts:
# puzzle pieces, contact

# description:
# In the input you will see blue puzzles pieces lying beneath a red shape with holes on its underside.
# To make the output, move the gray puzzle pieces upward so that they fit into the holes on the underside. They need to fully plug the holes.
# Finally change the grey puzzle pieces to be blue.

def main(input_grid, task=[]):
    # Plan:
    # 1. Detect the puzzle pieces and red thing
    # 2. Extract the sprites of each puzzle piece
    # 3. Try moving the puzzle pieces (but not so much that they collide with the red thing). You can translate but not rotate. Plug as much of the holes in the red thing as possible.
    # 4. Change the color of the puzzle pieces to blue (just change everything that's grey to blue)
    input_grid = np.asarray(input_grid)
    unique, counts = np.unique(input_grid, return_counts=True)
    # 개수 기준 내림차순 정렬
    sorted_indices = np.argsort(-counts)
    sorted_colors = unique[sorted_indices].tolist()

    bg = sorted_colors[0]
    red = sorted_colors[1]
    grey = sorted_colors[2]

    # 1. Separate puzzle pieces from the red object
    red_objects = detect_objects(grid=input_grid, colors=[red], monochromatic=True, connectivity=8, background=bg)
    assert len(red_objects) == 1, "There should be exactly one fully red object"
    red_object = red_objects[0]
    
    puzzle_pieces = detect_objects(grid=input_grid, colors=[grey], monochromatic=True, connectivity=4, background=bg)

    # 2. Extract sprites by cropping
    sprites = [ crop(piece, background=bg) for piece in puzzle_pieces ]

    # Output begins with just the red and then we add stuff to it
    output_grid = np.full_like(input_grid, bg)
    blit_object(output_grid, red_object, background=bg)

    # 3. Try moving the puzzle pieces, plugging as much of the red object is possible
    sprites_to_move = list(sprites)
    while sprites_to_move:
        possible_solutions = [ (x, y, sprite) for sprite in sprites_to_move 
                              for x in range(output_grid.shape[0] - sprite.shape[0] + 1)
                              for y in range(output_grid.shape[1] - sprite.shape[1] + 1) ]
        def score_solution(x, y, sprite):
            # The score is -inf if it collides with the red object
            # Otherwise it is the number of black pixels that are plugged by the sprite

            # Make a canvas by trying putting down the sprite
            test_canvas = np.full_like(output_grid, bg)
            blit_sprite(test_canvas, sprite, x, y)

            # Check for collision
            if collision(object1=test_canvas, object2=output_grid, background=bg):
                return float("-inf")
            
            # Count the number of black pixels that are plugged by the sprite, only counting those within the bounding box of the red object
            red_object_mask = red_object != bg
            test_object_mask = test_canvas != bg
            plugged_pixels = test_object_mask & ~red_object_mask & bounding_box_mask(red_object, background=bg)
            return np.sum(plugged_pixels)
        
        best_x, best_y, best_sprite = max(possible_solutions, key=lambda solution: score_solution(*solution))

        # Blit the sprite into the output grid
        blit_sprite(output_grid, best_sprite, best_x, best_y)

        # Remove the sprite from the list of sprites to move
        sprites_to_move = [ sprite for sprite in sprites_to_move if sprite is not best_sprite ]

    # 4. grey->blue
    output_grid[output_grid == grey] = Color.BLUE

    return output_grid

def generate_input():
    # Create a red rectangle and then chisel out some holes on the underside for some random sprites

    rectangle_width, rectangle_height = np.random.randint(15, 30), np.random.randint(5, 10)
    red_rectangle = np.full((rectangle_width, rectangle_height), Color.RED)

    # make the full grid such that it is wide enough for the rectangle but taller so that there is space for stuff underneath it
    width = rectangle_width
    height = np.random.randint(rectangle_height+5, 30)
    grid = np.full((width, height), bg)

    # put the red rectangle in the grid
    blit_sprite(grid, red_rectangle, 0, 0)

    # make the holes, each of which comes from a different puzzle piece (sprite)
    n_holes = np.random.randint(2, 4)
    for _ in range(n_holes):
        puzzle_piece = random_sprite([1,2,3,4], [1,2,3,4], connectivity=4, color_palette=[Color.GREY], background=bg)

        # find a possible place to chisel out a hole, which has to be a spot which would currently collide with the red rectangle
        possible_hole_locations = [ (x, y)
                                   for x in range(rectangle_width - puzzle_piece.shape[0] + 1)
                                   for y in range(rectangle_height - puzzle_piece.shape[1], rectangle_height) 
                                   if collision(object1=grid, object2=puzzle_piece, x2=x, y2=y) ]
        x, y = random.choice(possible_hole_locations)

        # chisel out the hole, which is everywhere 5the puzzle piece is grey
        for dx, dy in np.argwhere(puzzle_piece == Color.GREY):
            grid[x + dx, y + dy] = bg

        # find a place to put the puzzle piece in the bottom of the grid
        possible_puzzle_locations = [ (x, height - puzzle_piece.shape[1]) for x in range(width - puzzle_piece.shape[0] + 1) 
                                     if not contact(object1=grid, object2=puzzle_piece, x2=x, y2=height - puzzle_piece.shape[1]) ]
        x, y = random.choice(possible_puzzle_locations)
        blit_sprite(grid, puzzle_piece, x, y)
    
    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
