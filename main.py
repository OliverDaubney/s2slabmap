"""
Segment-to-Segment Label Mapping Script (S2SLabMap)

Purpose
-------
This is a script to map a set of labels from one image
to another by correctly orienting a smaller image onto
a larger image containing the same region as the smaller
image. This involves four steps:
1. Load small image, small_labels image and large image.
2. Orient the small image over the large image.  
3. Set bounding boxes around letter clusters.
4. Use the orientation of the small image to map clusters
to the large image.
5. Output large_labels image.

How To Use This Script
----------------------
1. Move tif files for large and small images into the
directory and label them 'large.tif' and 'small.tif'.
2. Move image file for labels into the directory and
label is 'small_inklabels.png'.
3. Open terminal locally within the directory and run
'python main.py'
4. Wait for the steps to complete (or any error message)
and collect output from the directory. The output will
be named 'large_inklabels.png'. There will also be an 
additional file called 'large_inklabels_papyrus.png'.

Dependencies
------------
Libraries: os, sys, collections, cv2 and numpy.
Local Files: support_classes.py and support_functions.py
"""

import os
import sys
import cv2
import numpy as np
import support_functions as sf
from support_classes import Mapper


def main():
    print('Welcome to the Segment-to-Segment Label Mapping Script.')
    
    # Set image filepaths.
    cwd = os.getcwd()
    small_path = os.path.join(cwd, 'small.tif')
    small_labels_path = os.path.join(cwd, 'small_inklabels.png')
    large_path = os.path.join(cwd, 'large.tif')

    # Check if the image filepaths exist.
    if not sf.check_image_paths(filepaths=[small_path, small_labels_path, large_path]):
        return

    # Load image files.
    small = sf.load_image(small_path, scale_tif=True)
    small_labels = sf.load_image(small_labels_path, scale_tif=False)
    large = sf.load_image(large_path, scale_tif=True)
    print('Loading Images: COMPLETE')

    # Check that the image shapes are compatible.
    shape_check = sf.check_image_shapes(small=small, small_labels=small_labels, large=large)
    if shape_check == 0:
        return
    elif shape_check == 1:
        small_labels = small_labels[0:small.shape[0], 0:small.shape[1]]

    # Run a first approximation to generally orient the labels.
    print('Orienting Images: running...(this can take a while!)', end='\r')
    orienter = Mapper(template_ratio=0.05)
    orienter.mount_images(large, small, small_labels)
    orienter.orient_images(settings=[0.0, 360.0, 0.0, 36], use_flip=True)
    small_labels = sf.rotate_image(small_labels, orienter.optimal_angle)
    small = sf.rotate_image(small, orienter.optimal_angle)
    if orienter.optimal_flip == 1:
        small_labels = cv2.flip(small_labels, 1)
        small = cv2.flip(small, 1)
    # Update the location of the top left hand corner of the labels inside the large segment.
    off_y, off_x = sf.get_offset(small_labels, orienter.get_min_side_ratio(small_labels))
    loc = (orienter.optimal_loc[1]-off_y, orienter.optimal_loc[0]-off_x)
    print('Orienting Images: COMPLETE')

    # Find letter clusters using a modified breadth first search algorithm.
    print('Letter Cluster Detection: running...', end='\r')
    boundaries = sf.get_letter_boundries(label_image=small_labels, split_boxes=True)
    print(f'Letter Cluster Detection: {len(boundaries)} Letter Clusters')

    # Create a match handler, orient images and map labels from small to large.
    labels = np.zeros(large.shape)
    for index, box in enumerate(boundaries):
        print(f'Mapping Labels: {index}', end='\r')
        box.create_contents(image=small)
        if np.prod(box.contents.shape) < 15000:
            lab_mapper = Mapper(template_ratio=0.7)
        else:
            lab_mapper = Mapper(template_ratio=0.4)
        Y_MIN, Y_MAX, X_MIN, X_MAX = sf.offsets(large, box.contents, (box.y_min+loc[0], box.x_min+loc[1]))
        large_section = large[Y_MIN:Y_MAX, X_MIN:X_MAX]
        lab_mapper.mount_images(large_section, box.contents, box.mask)
        lab_mapper.orient_images(settings=[-10.0, 10.0, 0.0, 7], use_flip=False)
        lab_mapper.map_labels(np.zeros(large_section.shape))
        temp = labels[Y_MIN:Y_MAX, X_MIN:X_MAX]
        labels[Y_MIN:Y_MAX, X_MIN:X_MAX] = np.where((lab_mapper.large_labels > 0), lab_mapper.large_labels, temp)
        sf.save_image(
            filepath=os.path.join(cwd, 'large_inklabels.png'),
            image=labels
        )
    print('Mapping Labels: COMPLETE')

    large[large < labels] = 0
    sf.save_image(
        filepath=os.path.join(cwd, 'large_inklabels_papyrus.png'),
        image=large
    )


if __name__ == '__main__':
    main()
    sys.exit('Goodbye')
