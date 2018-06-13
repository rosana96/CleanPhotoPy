# USAGE
# python3 main.py --first images/r3.jpg --second images/r4.jpg

import imutils

from ImageProcessor import *
# import the necessary packages
from utils.FrameExtractor import FrameExtractor

WIDTH = 500


def useSeparateImages():
    first = "images/cables/cb1.jpg"
    second = "images/cables/cb2.jpg"
    third = "images/cables/cb3.jpg"
    fourth = "images/cables/cb4.jpg"
    imageA = cv2.imread(first)
    imageB = cv2.imread(second)
    imageC = cv2.imread(third)
    imageD = cv2.imread(fourth)
    imageA = imutils.resize(imageA, width=WIDTH)
    imageB = imutils.resize(imageB, width=WIDTH)
    imageC = imutils.resize(imageC, width=WIDTH)
    imageD = imutils.resize(imageD, width=WIDTH)

    images = [imageA, imageB, imageC, imageD]
    return images


def useVideo():
    fe = FrameExtractor('videos/forest180.mp4')
    imgs = fe.extractFrames()
    images = []
    for i in imgs:
        i = imutils.resize(i, width=WIDTH)
        images.append(i)

    return images


# images = useVideo()
images = useSeparateImages()

blockDim=9
imgProc = MovingCameraImageProcessor(images, blockDim)
cleanImage = imgProc.reconstructCleanImage()
imagesFolder = "images/results"
filename = imagesFolder + "/image_" + str(blockDim) + ".jpg"
cv2.imwrite(filename, cleanImage)

cv2.imshow("clean"+str(blockDim), cleanImage)

# for i in range (4,32,4):
#     blockDim = i
#     imgProc = MovingCameraImageProcessor(images, blockDim)
#     cleanImage = imgProc.reconstructCleanImage()
#     cv2.imshow("clean"+str(i), cleanImage)

# blur = cv2.blur(cleanImage, (5, 5))
# cv2.imshow("blurred", blur)
cv2.waitKey(0)
