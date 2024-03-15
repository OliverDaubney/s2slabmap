"""
The is a support file for the program:
Segment-to-Segment Label Mapping Script (S2SLabMap)

This file contains:
- Bounding_Box class.
- Matcher class.
- Mapper class.

Dependencies
------------
Libraries: cv2 and numpy.
Local Files: support_functions.py (included in repo).
"""

import cv2
import numpy as np
import support_functions as sf


class Bounding_Box:
    """
        Accepts a position in the format of (x, y)  
        and maintains an updated boundary set of  
        values. This class also maintains an array  
        of contents and a mask for the contents.
    """
    def __init__(self, position=(0, 0)):
        self.x_min = position[0]
        self.x_max = position[0]
        self.y_min = position[1]
        self.y_max = position[1]

    def update_boundaries(self, position):
        if position[0] < self.x_min:
            self.x_min = position[0]
        elif position[0] > self.x_max:
            self.x_max = position[0]
        if position[1] < self.y_min:
            self.y_min = position[1]
        elif position[1] > self.y_max:
            self.y_max = position[1]

    def create_contents(self, image):
        self.contents = image[self.y_min:self.y_max+1, self.x_min:self.x_max+1]

    def create_mask(self, positions):
        self.mask = np.zeros((self.y_max-self.y_min+1, self.x_max-self.x_min+1))
        for position in positions:
            self.mask[position[1]-self.y_min, position[0]-self.x_min] = 255


class Matcher:
    """
        Sets up a matching method and logs the best  
        angle, flip state and location for matches  
        between an image and a large image.
    """
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
            self.check_match_and_update(value=1-min_val, angle=angle, location=min_loc, flipped=flipped)
        else:
            self.check_match_and_update(value=max_val, angle=angle, location=max_loc, flipped=flipped)

    def check_match_and_update(self, value, angle, location, flipped):
        if value > self.value:
            self.value, self.angle, self.loc = value, angle, location
            if flipped:
                self.flipped = 1
            else:
                self.flipped = 0


class Mapper:
    """
        Manages the mounting of large, small and  
        small labels images. The small image is  
        oriented to overlay with the large image  
        and labels from the small labels image are  
        mapped to generate a large labels image.
    """
    def __init__(self, methods=[cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED], template_ratio=0.3):
        self.match_makers = [Matcher(method) for method in methods]
        self.template_ratio = template_ratio
        self.mounted = False
        self.oriented = False
        self.optimal_angle = 0
        self.optimal_loc = (0, 0)
        self.optimal_flip = 0

    def mount_images(self, large, small, small_labels):
        self.large = large
        self.small = small
        self.small_labels = small_labels
        self.mounted = True

    def orient_images(self, settings=[0, 360, 0, 36], use_flip=False):
        if not self.mounted:
            print("Error - No images have been mounted to the match handler.")
            return
        self.optimal_angle, self.optimal_flip, self.optimal_loc = self.find_angle_and_flip(
            range_min=settings[0],
            range_max=settings[1],
            previous_angle=settings[2],
            num_steps=settings[3],
            use_flip=use_flip
        )
        self.oriented = True

    def map_labels(self, label_map=None):
        if not self.oriented:
            print("Error - The images have not been oriented.")
            return

        labels = sf.rotate_image(self.small_labels, self.optimal_angle)
        if self.optimal_flip == 1:
            labels = cv2.flip(labels, 1)
        off_y, off_x = sf.get_offset(labels, self.get_min_side_ratio(labels))
        loc = (self.optimal_loc[1]-off_y, self.optimal_loc[0]-off_x)

        self.large_labels = label_map
        labels, loc = sf.crop_image(labels, self.large_labels, loc)
        self.large_labels[loc[0]:loc[0]+labels.shape[0], loc[1]:loc[1]+labels.shape[1]] = labels

    def find_angle_and_flip(self, range_min=0, range_max=360, previous_angle=0, num_steps=18, use_flip=False):
        # Got to be a cleaner way of finding the optimal angle.
        step_size = (range_max - range_min) / num_steps
        steps = np.arange(range_min, range_max, step_size)
        for step in steps:
            rotated = sf.rotate_image(self.small, step)
            template_size = self.get_min_side_ratio(rotated)
            template = sf.create_template(rotated, template_size, flip_img=False)
            for match_maker in self.match_makers:
                match_maker.match(template, self.large, angle=step, flipped=False)
            if use_flip:
                template = sf.create_template(rotated, template_size, flip_img=True)
                for match_maker in self.match_makers:
                    match_maker.match(template, self.large, angle=step, flipped=True)

        best_angle = sf.find_most_common([m.angle for m in self.match_makers])
        if abs(best_angle-previous_angle) < 0.1:
            best_flip = sf.find_most_common([m.flipped for m in self.match_makers])
            best_loc = sf.find_most_common([m.loc for m in self.match_makers])
            return best_angle, best_flip, best_loc
        else:
            return self.find_angle_and_flip(best_angle-step_size/2, best_angle+step_size/2, best_angle, 7, use_flip)

    def get_min_side_ratio(self, image):
        min_side = min(image.shape)
        return int(self.template_ratio * min_side)
