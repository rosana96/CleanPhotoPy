# USAGE
# python3 main.py --first images/r3.jpg --second images/r4.jpg

import imutils

from ImageProcessor import *
# import the necessary packages
from utils.FrameExtractor import FrameExtractor

WIDTH = 600


def useSeparateImages():
    images=[]
    filename="images/cables/cb"
    for i in range (1,5):
        fullname=filename+str(i)+".jpg"
        image=cv2.imread(fullname)
        image = imutils.resize(image, width=WIDTH)
        images.append(image)
    return images


def useVideo():
    fe = FrameExtractor('videos/vid7.mp4')
    imgs = fe.extractFrames()
    images = []
    for i in imgs:
        i = imutils.resize(i, width=WIDTH)
        images.append(i)

    return images


images = useVideo()
# images = useSeparateImages()

blockDim=4
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


# stalp1 = "images/cables/palat1.jpg"
# stalp2 = "images/cables/palat2.jpg"
# imageA = cv2.imread(stalp1)
# imageB = cv2.imread(stalp2)
#
# imageA = imutils.resize(imageA, width=WIDTH)
# imageB = imutils.resize(imageB, width=WIDTH)
# stalpi=[imageA,imageB]
#
# matcher = Matcher.Matcher()
# result, vis = matcher.getHomography(stalpi)
# cv2.imshow("result", result)
# cv2.imshow("vis", vis)
# cv2.imwrite("images/Matches_Palat.jpg",vis)

cv2.waitKey(0)


#TODO sa nu ia in considerare matches din prima treime de imagine