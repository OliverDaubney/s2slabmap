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
3. Use the orientation of the small image to map labels
to the large image.
4. Output large_labels image.

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
additional file called 'large_inklabels_test.png'.

Dependencies
------------
Libraries: os, sys, collections, cv2 and numpy.
Local Files: support_file.py (included in repo).
"""

import os
import sys
import support_file as sf


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
    
    # Create a match handler, orient images and map labels from small to large.
    match_handler = sf.Match_Handler(template_size=200)
    match_handler.mount_images(large, small, small_labels, padding=True)
    match_handler.orient_images()
    match_handler.map_labels()
    large[large < match_handler.large_labels] = 0
    print('Mapping Labels: COMPLETE')
    
    # Save outputs.
    sf.save_image(
        filepath=os.path.join(cwd, 'large_inklabels.png'),
        image=match_handler.large_labels
    )
    sf.save_image(
        filepath=os.path.join(cwd, 'large_inklabels_test.png'),
        image=large
    )
    print('Saving Output: COMPLETE')


if __name__ == '__main__':
    main()
    sys.exit('Goodbye')
