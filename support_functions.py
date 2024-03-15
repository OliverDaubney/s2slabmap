"""
The is a support file for the program:
Segment-to-Segment Label Mapping Script (S2SLabMap)

This file contains:
- functions to support bfs algorithm for feature separation.
- functions to support geometric operations.
- functions to support data structure operations.
- functions to support image checking and handling.

Dependencies
------------
Libraries: os, cv2, numpy, and collections.
Local Files: support_classes.py (included in repo).
"""

import os
import cv2
import numpy as np
from collections import Counter, deque
from support_classes import Bounding_Box


# Functions to support bfs algorithm for feature separation.
def bfs(start, image, grid):
    box = Bounding_Box(position=(start[1], start[0]))
    positions = []
    cells_to_check = deque([start])
    while cells_to_check:
        current = cells_to_check.popleft()
        grid[current[0], current[1]] = False
        cells_to_check = fetch_neighbours(current, cells_to_check, grid)
        box.update_boundaries(position=(current[1], current[0]))
        positions.append((current[1], current[0]))
    box.create_mask(positions)
    return grid, box

def fetch_neighbours(current, cells_to_check, grid):
    if current[0] > 0 and grid[current[0]-1, current[1]] == True and (current[0]-1, current[1]) not in cells_to_check:
        cells_to_check.append((current[0]-1, current[1]))
    if current[0] < grid.shape[0]-1 and grid[current[0]+1, current[1]] == True and (current[0]+1, current[1]) not in cells_to_check:
        cells_to_check.append((current[0]+1, current[1]))
    if current[1] > 0 and grid[current[0], current[1]-1] == True and (current[0], current[1]-1) not in cells_to_check:
        cells_to_check.append((current[0], current[1]-1))
    if current[1] < grid.shape[1]-1 and grid[current[0], current[1]+1] == True and (current[0], current[1]+1) not in cells_to_check:
        cells_to_check.append((current[0], current[1]+1))
    return cells_to_check

def get_letter_boundries(label_image, split_boxes=True):
    grid_false = np.full(label_image.shape, False)
    grid_true = np.full(label_image.shape, True)
    grid = np.where((label_image != 0), grid_true, grid_false)

    boxes = []
    while grid.any():
        i, j =  np.unravel_index(np.argmax(grid, axis=None), grid.shape)
        grid, box = bfs([i, j], label_image, grid)
        if box.x_max-box.x_min > 20 and box.y_max-box.y_min > 20:
            ###Need to split the box up if it is too long...
            ###NOT AN IDEAL SOLUTION - NEED TO WORK ON LETTER RECOGNITION...
            if split_boxes and box.mask.shape[1] > 500:
                for i in range(0, (box.mask.shape[1]//500)):
                    new = Bounding_Box()
                    new.x_min = box.x_min
                    new.x_max = box.x_min + 500
                    new.y_min = box.y_min
                    new.y_max = box.y_max
                    new.mask = box.mask[:, 0:500]
                    box.x_min += 500
                    box.mask = box.mask[:, 500:]
                    boxes.append(new)
            boxes.append(box)      
    return boxes


# Functions to support geometric operations.
def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def get_offset(image, size):
    # get left offset for a centered square with sides equal to size.
    off_y, off_x = image.shape[0]//2 - size//2, image.shape[1]//2 - size//2
    return off_y, off_x

def crop_image(small, large, loc):
    if loc[0]+small.shape[0] > large.shape[0]:
        small = small[:-(loc[0]+small.shape[0]-large.shape[0])]
    if loc[0] < 0:
        small = small[abs(loc[0]):]
        loc = (0, loc[1])
    if loc[1]+small.shape[1] > large.shape[1]:
        small = small[:, :-(loc[1]+small.shape[1]-large.shape[1])]
    if loc[1] < 0:
        small = small[:, abs(loc[1]):]
        loc = (loc[0], 0)
    return small, loc

def create_template(image, size, flip_img=False):
    off_y, off_x = get_offset(image, size)
    image_section = image[off_y:off_y+size, off_x:off_x+size]
    if flip_img:
        return cv2.flip(image_section, 1)
    else:
        return image_section

def offsets(large, small, loc):
    MIN_OFFSET_Y, MIN_OFFSET_X = 0, 0
    MAX_OFFSET_Y, MAX_OFFSET_X = large.shape
    if loc[0] > 200:
        MIN_OFFSET_Y = loc[0] - 200
    if loc[0] + small.shape[0] + 400 < large.shape[0]:
        MAX_OFFSET_Y = loc[0] + small.shape[0] + 400
    if loc[1] > 200:
        MIN_OFFSET_X = loc[1] - 200
    if loc[1] + small.shape[1] + 400 < large.shape[1]:
        MAX_OFFSET_X = loc[1] + small.shape[1] + 400
    return MIN_OFFSET_Y, MAX_OFFSET_Y, MIN_OFFSET_X, MAX_OFFSET_X


# Function to support data structure operations.
def find_most_common(values):
    counter = Counter(values)
    return counter.most_common(1)[0][0] 


# Functions to support general image checking and handling operations.
def check_image_paths(filepaths):
    for filepath in filepaths:
        if not os.path.isfile(filepath):
            print(f'Error - Source file ({filepath}) could not be found.')
            return False
    return True

def check_image_shapes(small, small_labels, large):
    # if small.shape[0]*small.shape[1] > large.shape[0]*large.shape[1]:
    #     print('Error - The small image is larger than the large image.')
    #     return 0
    if small_labels.shape[0] != small.shape[0] or small_labels.shape[1] != small.shape[1]:
        if small_labels.shape[0] % 256 == 0 and small_labels.shape[1] % 256 == 0:
            return 1
        else:
            print("Error - The small image shape is not the same as the labels shape.")
            return 0
    return 2

def load_image(filepath, scale_tif=False, grayscale=True):
    if scale_tif:
        image = cv2.convertScaleAbs(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), alpha=(255.0/65535.0))
    else:
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # Initially force single channel grayscale as output.
    if image.shape[-1] == 3 and grayscale:
        return np.average(image, axis=2)
    return image

def save_image(filepath, image):
    cv2.imwrite(filepath, image)
