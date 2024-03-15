# Segment-to-Segment Label Mapping

## Purpose
This is a script to map a set of labels from one image to another by correctly orienting a smaller image onto
a larger image containing the same region as the smaller image. This involves five steps:  
1. Load small image, small_labels image and large image.
2. Orient the small image over the large image.
3. Set bounding boxes around letter clusters.
4. Use the orientation of the small image to map clusters to the large image.
5. Output large_labels image.  

Note: The clusters will be mapped with the same geometry, which can result in historic artefacts being carried forward.

## How To Use This Script
1. Move tif files for large and small images into the directory and label them 'large.tif' and 'small.tif'.
2. Move image file for labels into the directory and label is 'small_inklabels.png'.
3. Open terminal locally within the directory and run 'python main.py'
4. Wait for the steps to complete (or any error message) and collect output from the directory. The output will
be named 'large_inklabels.png'. There will also be an additional file called 'large_inklabels_test.png'.

## Dependencies
Libraries: os, sys, collections, cv2 and numpy.  
Local Files: support_functions.py and support_classes.py (included in repo).

## Pipeline
The overall process pipeline.  
![Process Pipeline](https://github.com/OliverDaubney/s2slabmap/blob/main/images/pipeline.png)  

