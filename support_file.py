"""
The is a support file for the program:
Segment-to-Segment Label Mapping Script (S2SLabMap)

This file contains:
- class Matcher.
- class Match_Handler.
- functions to support geometric operations.
- functions to support data structure operations.
- functions to support image checking and handling.

Dependencies
------------
Libraries: os, collections, cv2 and numpy.
"""

import os
import cv2
import numpy as np
from collections import Counter


class Matcher:
    def __init__(self, method):
        self.method = method
        self.value = 0
        self.angle = 0
        self.loc = (0, 0)
        self.flipped = 0

    def match(self, image, large, angle, flipped=False):
        result = cv2.matchTemplate(large, image, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            self.check_match_and_update(value=1/min_val, angle=angle, location=min_loc, flipped=flipped)
        else:
            self.check_match_and_update(value=max_val, angle=angle, location=max_loc, flipped=flipped)

    def check_match_and_update(self, value, angle, location, flipped):
        if value > self.value:
            self.value, self.angle, self.loc = value, angle, location
            if flipped:
                self.flipped = 1
            else:
                self.flipped = 0


class Match_Handler:
    def __init__(self, methods=[cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED], template_size=500):
        self.match_makers = [Matcher(method) for method in methods]
        self.template_size = template_size
        self.mounted = False
        self.oriented = False
        self.optimal_angle = 0
        self.optimal_loc = (0, 0)
        self.optimal_flip = 0

    def mount_images(self, large, small, small_labels, padding=True):
        if padding:
            self.small = pad_to_square(small)
            self.small_labels = pad_to_square(small_labels)
        else:
            self.small = small
            self.small_labels = small_labels
        self.large = large
        self.mounted = True

    def orient_images(self):
        if not self.mounted:
            print("Error - No images have been mounted to the match handler.")
            return
        self.optimal_angle, self.optimal_flip, self.optimal_loc = self.find_angle_and_flip(num_steps=36)
        self.oriented = True

    def map_labels(self):
        if not self.oriented:
            print("Error - The images have not been oriented.")
            return

        small_labels = rotate_image(self.small_labels, self.optimal_angle)
        if self.optimal_flip == 1:
            small_labels = cv2.flip(small_labels, 1)
        off_y, off_x = get_offset(self.small_labels, self.template_size)
        loc = (self.optimal_loc[1]-off_y, self.optimal_loc[0]-off_x)

        self.large_labels = np.zeros(self.large.shape)
        small_labels, loc = crop_image(small_labels, self.large_labels, loc)
        self.large_labels[loc[0]:loc[0]+small_labels.shape[0], loc[1]:loc[1]+small_labels.shape[1]] = small_labels

    def find_angle_and_flip(self, range_min=0, range_max=360, previous_angle=0, num_steps=18, count=0):
        # Got to be a cleaner way of finding the optimal angle.
        step_size = (range_max - range_min) / num_steps
        steps = np.arange(range_min, range_max, step_size)
        for step in steps:
            print(f'Orienting Images: {(count/0.57):.2f}%', end='\r')
            rotated = rotate_image(self.small, step)
            template = self.create_template(rotated, self.template_size, flip_img=False)
            for match_maker in self.match_makers:
                match_maker.match(template, self.large, angle=step, flipped=False)
            template = self.create_template(rotated, self.template_size, flip_img=True)
            for match_maker in self.match_makers:
                match_maker.match(template, self.large, angle=step, flipped=True)
            count += 1

        best_angle = find_most_common([m.angle for m in self.match_makers])
        if abs(best_angle-previous_angle) < 0.1:
            best_flip = find_most_common([m.flipped for m in self.match_makers])
            best_loc = find_most_common([m.loc for m in self.match_makers])
            print(f'Orienting Images: COMPLETE')
            return best_angle, best_flip, best_loc
        else:
            return self.find_angle_and_flip(best_angle-step_size/2, best_angle+step_size/2, best_angle, 7, count)

    def create_template(self, image, size, flip_img=False):
        off_y, off_x = get_offset(image, size)
        image_section = image[off_y:off_y+size, off_x:off_x+size]
        if flip_img:
            return cv2.flip(image_section, 1)
        else:
            return image_section


# Functions to support geometric operations.
def pad_to_square(image):
    side = int((image.shape[0]**2 + image.shape[1]**2)**0.5)
    temp = np.zeros((side, side))
    y_pos, x_pos = (side-image.shape[0])//2, (side-image.shape[1])//2
    temp[y_pos:y_pos+image.shape[0], x_pos:x_pos+image.shape[1]] = image
    return temp.astype('uint8')

def rotate_image(image, angle):
    image_center = tuple(((np.array(image.shape[1::-1])-1) / 2))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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
    if small.shape[0]*small.shape[1] > large.shape[0]*large.shape[1]:
        print('Error - The small image is larger than the large image.')
        return 0
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
