# USAGE
# python3 main.py --first images/r3.jpg --second images/r4.jpg

import imutils

from ImageProcessor import *
# import the necessary packages
from utils.FrameExtractor import FrameExtractor

WIDTH = 750
FRAMES_PER_SECOND = 1


def useSeparateImages():
    images = []
    filename = "images/cables/palat"
    for i in range(1, 5):
        fullname = filename + str(i) + ".jpg"
        image = cv2.imread(fullname)
        image = imutils.resize(image, width=WIDTH)
        images.append(image)
    return images


def useVideo():
    fe = FrameExtractor('videos/vid15.mp4', FRAMES_PER_SECOND)
    imgs = fe.extractFrames()
    images = []
    k = 0
    for i in imgs:
        i = imutils.resize(i, width=WIDTH)
        images.append(i)
        # cv2.imshow("im"+str(k),i)
        k = k + 1
    return images


images = useVideo()
# images = useSeparateImages()

blockDim = 16
imgProc = MovingCameraImageProcessor(images, blockDim)
cleanImage = imgProc.reconstructCleanImage()
imagesFolder = "images/results"
filename = imagesFolder + "/img_" + str(blockDim) + "_" + str(WIDTH) + "_" + str(FRAMES_PER_SECOND) + ".jpg"
cv2.imwrite(filename, cleanImage)

cv2.imshow("clean" + str(blockDim), cleanImage)

# for i in range (4,32,4):
#     blockDim = i
#     imgProc = MovingCameraImageProcessor(images, blockDim)
#     cleanImage = imgProc.reconstructCleanImage()
#     cv2.imshow("clean"+str(i), cleanImage)

# blur = cv2.blur(cleanImage, (5, 5))
# cv2.imshow("blurred", blur)


# stalp1 = "images/cables/palat1.jpg"
# stalp2 = "images/cables/palat2.jpg"
# imageA = cv2.imread(stalp1)
# imageB = cv2.imread(stalp2)
#
# imageA = imutils.resize(imageA, width=WIDTH)
# imageB = imutils.resize(imageB, width=WIDTH)
# imgs=[images[6], images[3]]
#
# matcher = Matcher.Matcher()
# result, vis = matcher.getHomography(imgs)
# # cv2.imshow("result", result)
# cv2.imshow("vis", vis)
# cv2.imwrite("images/Matches_coregrafie1.jpg",vis)

cv2.waitKey(0)


# TODO sa nu ia in considerare matches din prima treime de imagine
