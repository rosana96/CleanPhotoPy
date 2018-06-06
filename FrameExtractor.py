import math

import cv2
import sys
import skvideo.io


class FrameExtractor:
    def __init__(self, videoFile):
        self._videoFile = videoFile
        self._frames = []

    def extractFrames(self):
        videoFile = self._videoFile
        imagesFolder = "images/frames"
        cap = cv2.VideoCapture(videoFile)
        frameRate = cap.get(5)  # frame rate
        while (cap.isOpened() or True):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
                cv2.imwrite(filename, frame)
        cap.release()



fe = FrameExtractor('v2.mp4')
fe.extractFrames()
