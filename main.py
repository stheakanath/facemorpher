# Kuriakose Sony Theakanath
# Face Morphing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import constant
from PIL import Image
from scipy import misc
import argparse
from pylab import arange, plot, sin, ginput, show
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from images2gif import writeGif

# Part 1 - Allows user to select points on the supplied image.
def selectPoints(im1_path, im2_path):
	im = Image.open(im1_path)
	plt.imshow(im)
	counter, f_points = constant.TOTAL_FEATURE, []
	while counter != 0:
		print "Click on screen!"
		x = ginput(1)
		counter -= 1
		f_points.append([x[0][0], x[0][1]])
		plt.scatter(x[0][0], x[0][1])
		plt.draw()
		print("Clicked point at ", x, " | Clicks left: ", counter)
	plt.show()
	second_points = drag_control_points(mpimg.imread(im2_path), np.array(f_points))
	
	intermediate_feature = interpolatePts(combinePoints(f_points, second_points))
	frames = combineImages(intermediate_feature, constant.TRIANGLES, im1_path, im2_path)
	frames.extend(frames[::-1])
	# otherone = [cv2.cvtColor(items, cv2.COLOR_RGB2BGR) for items in frames]
	# writeGif("lol.GIF", otherone, duration=0.07)
	while True:
		for i in range (0, len(frames)): 
			f = frames[i]
			cv2.waitKey(20) 
			cv2.imshow("Cameras",f) 
			cv2.waitKey(20)

# Step 2 - Creates a triangulation from the points given
def interpolatePts(features):
	frame, middle = [(constant.RATIO * i) for i in xrange(0, 22)], []
	for r in xrange(0, len(frame)):
		middle.append([(pair[0][0] * (1 - frame[r]) + pair[1][0] * frame[r], pair[0][1] * (1-frame[r]) + pair[1][1] * frame[r]) for pair in features] + constant.CORNERS)
	return middle

# Step 3, takes the features and warps it according to the triangles.
def warpImage(orig, features, diang, src):
	image = cv2.imread(src)
	masked_image = np.zeros(image.shape, dtype=np.uint8)
	for t in diang:
		mask = np.zeros(image.shape, dtype=np.uint8)
		cv2.fillPoly(mask, np.array([[features[t[0]], features[t[1]], features[t[2]]]], dtype=np.int32), (255, 255, 255))
		masked_image = cv2.bitwise_or(masked_image, cv2.bitwise_and(cv2.warpAffine(image, cv2.getAffineTransform(np.float32([orig[t[0]], orig[t[1]], orig[t[2]]]), np.float32([features[t[0]], features[t[1]], features[t[2]]])), (image.shape[1],image.shape[0])), mask))
	return masked_image

# Step 4, takes the warped images, and warps it, creating a video frame for viewing.
def combineImages(features, diag, path1, path2):
	frames = []
	for i in xrange(0, 22):
		frames.append(cv2.addWeighted(warpImage(features[0], features[i], diag, path1), 1 - constant.RATIO * i, warpImage(features[21], features[i], diag, path2), constant.RATIO * i, 0))
	return frames

# Sub-process - calculates the average face with a provided folder
def averageFace(path):
	TOTAL_NUM = 100
	w, h, arr = Image.open(path + "/1a.jpg").size, np.zeros((h,w,3),np.float)
	while TOTAL_NUM != 0:
		arr = arr + np.array(Image.open(path + "/" + str(TOTAL_NUM) + "a.jpg"), dtype=np.float) / 100
		TOTAL_NUM -= 1
	arr = np.array(np.round(arr), dtype=np.uint8)
	Image.fromarray(arr, mode="RGB").save("average_ " + path + ".png")

# Sub-process - takes an image, allows a user to select points, and exports to .dat file
def exportShape(path):
	plt.imshow(Image.open(path))
	counter, f_points = constant.TOTAL_FEATURE, []
	while counter != 0:
		print "Click on screen!"
		x = ginput(1)
		counter -= 1
		f_points.append([x[0][0], x[0][1]])
		plt.scatter(x[0][0], x[0][1])
		plt.draw()
		print("Clicked point at ", x, " | Clicks left: ", counter)
	plt.show()
	np.savetxt('shape.dat', f_points)
	return f_points

# For selection of the second image - borrowed from Piazza with a few modifications
def drag_control_points(img, cpts):
    cpts = cpts.copy()
    scale = (img.shape[0]**2 + img.shape[1]**2)**0.5/20
    fh = plt.figure('Close window to terminate')
    ah = fh.add_subplot(111)
    ah.imshow(img, cmap='gray')
    temp = ah.axis()
    ah.set_xlim(temp[0:2])
    ah.set_ylim(temp[2:4])
    lh = [None]
    lh[0] = ah.plot(cpts[:,0], cpts[:,1], 'g.')[0]

    idx = [None]
    figure_exist = [True]

    def on_press(event):
        diff = np.abs(np.array([[event.xdata, event.ydata]]) - cpts).sum(axis=(1,))
        idx[0] = np.argmin(diff)
        if diff[idx[0]] > scale:
            idx[0] = None
        else:
            temp_cpts = np.delete(cpts, idx[0], axis=0)
            lh[0].remove()
            lh[0] = ah.plot(temp_cpts[:,0], temp_cpts[:,1], 'g.')[0]
            fh.canvas.draw()

    def on_release(event):
        if idx[0] != None:
            cpts[idx[0], 0] = event.xdata
            cpts[idx[0], 1] = event.ydata
            lh[0].remove()
            lh[0] = ah.plot(cpts[:,0], cpts[:,1], 'g.')[0]
            fh.canvas.draw()

    def handle_close(event):
        figure_exist[0] = False

    fh.canvas.mpl_connect('close_event', handle_close)
    fh.canvas.mpl_connect('button_press_event', on_press)
    fh.canvas.mpl_connect('button_release_event', on_release)
    fh.show()
    while figure_exist[0]:
        plt.waitforbuttonpress()
    return cpts

# Helper function to combine points from image 1 and image 2
def combinePoints(pt1, pt2):
	super_array = []
	for coor1, coor2 in zip(pt1, pt2):
		super_array.append([tuple(coor1), tuple(coor2.tolist())])
	return super_array

# Main Function Calls
# averageFace("frontimages")
selectPoints(sys.argv[1], sys.argv[2])
