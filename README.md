# Segment-to-Segment Label Mapping

## Purpose
This is a script to map a set of labels from one image  
to another by correctly orienting a smaller image onto  
a larger image containing the same region as the smaller  
image. This involves four steps:  
1. Load small image, small_labels image and large image.
2. Orient the small image over the large image.
3. Use the orientation of the small image to map labels to the large image.
4. Output large_labels image.  

Note: The labels will be mapped with the same geometry.  
Which can result in historic artefacts being carried forward.

## How To Use This Script
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

## Dependencies
Libraries: os, sys, collections, cv2 and numpy.  
Local Files: support_file.py (included in repo).

## Tests
Test 1: Labels from 20231016151000 mapped to 20231016151002.  
Test 2: Labels from 20231005123333 mapped to 20231005123336.  
Test 3: Labels from 20230827161846 mapped to 20230827161847.  


