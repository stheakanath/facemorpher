Kuriakose Sony Theakanath
Face Morphing
README.md

NOT COMPLETE.

================
Running the Code
================
python main.py <image1> <image2>

Note you'll need a dependency on cv2 for the some of the helper functions.

Also the selection of the points is dependent on the order. If you select the points in an incorrect order, you may not get a good morph!

The code will first ask you to select 60 points on the first image. Once you select all 60 points then you can hit the red bubble to move on. The second image that shows allows you to move the points so that you can select the defining features of the second image. Again, hit the red bubble (close button) to move on. 

===================
Explanation of Code
===================
Now let's go through all of the functions in the readme. Note that this is in chronological order of how the functions are called when the user runs the code. 

The main function in the code is selectPoints(im1_path, im2_path). selectPoints takes 2 images and shows the image using matplotlib. The user then has to select 60 points and then they can move on. Code wise you are selecting the defining features of the first person so that the triangulation formula will work correctly. The code then relays these same points on the second image and then the user can move the points based on the second person's features. The function that handles this call is:

drag_control_points(img, cpts). drag_control_points(img, cpts) then calls combinePoints(pt1, pt2) which combines both of the points selected by the user into a tuple array so that the actual algorithm can process it. 

interpolatePts(features) then takes the array containing all of the tuples and then creates a Delaunay triangulation using the cv2 API. 

warpImage(orig, features, diang, src) is the the third function that is called after interpolatePts. warpImage takes the original image, the feature points created, the diagnoals that were returned by interpolatePts. warpImage then returns an image that is warped according to the triangles defined by the Delaunay triangulation formula. 

def combineImages(features, diag, path1, path2) takes the warped image from the previous function and then creates 22 frames of how the image is being warped. We compute a ratio based on the magic number 22 and compute the warp based on that ratio. 

After combineImages is called, we display it on the main screen and we are done!

Helper Functions Explanation
----------------------------
averageFace(path) - Average face takes a folder and returns the average face from the population. It adds all of the faces together and divides it by the amount of images in that certain folder. 

exportShape(path) - Allows the user to select points on a defined image and then exports a .dat file so that the user can read it in again. 