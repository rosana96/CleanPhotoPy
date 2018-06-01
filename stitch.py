# USAGE
# python3 stitch.py --first images/r3.jpg --second images/r4.jpg

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2


def cmdLineRun():
	global imageA, imageB
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True,
					help="path to the first image")
	ap.add_argument("-s", "--second", required=True,
					help="path to the second image")
	args = vars(ap.parse_args())
	imageA = cv2.imread(args["first"])
	imageB = cv2.imread(args["second"])


# construct the argument parse and parse the arguments
# cmdLineRun()

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

first="images/stalp1.jpg"
second="images/stalp2.jpg"
imageA = cv2.imread(first)
imageB = cv2.imread(second)
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)