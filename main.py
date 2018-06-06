# USAGE
# python3 main.py --first images/r3.jpg --second images/r4.jpg

import cv2
import imutils

# import the necessary packages
from pyimagesearch.ImageProcessor import ImageProcessor
from pyimagesearch.panorama import Stitcher

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

first = "images/stalp1.jpg"
second = "images/stalp2.jpg"
third = "images/cables/cab3.jpg"
imageA = cv2.imread(first)
imageB = cv2.imread(second)
imageC = cv2.imread(third)

# todo do I need this?
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
imageC = imutils.resize(imageC, width=400)

images = [imageA, imageB]
homographies = []

stitcher = Stitcher()

for i in range(0, len(images)-1):
    h= stitcher.getHomography(images[i:i + 2])
    homographies.append(h)

imgProc = ImageProcessor(images, homographies)
# imgProc.computeCleanImage()


# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
# cv2.imshow("Result", result)
cv2.waitKey(0)
