import math

import cv2


class FrameExtractor:
    def __init__(self, videoFile, framesPerSecond=2):
        self._videoFile = videoFile
        self._frames = []
        self._framesPerSecond = framesPerSecond

    def extractFrames(self):
        imagesFolder = "images/frames"
        images = []
        cap = cv2.VideoCapture(self._videoFile)
        cap.open(self._videoFile)
        frameRate = cap.get(5)
        while (cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if (ret != True):
                break
            if frameId % (math.floor(frameRate) // self._framesPerSecond) == 0:
                filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
                cv2.imwrite(filename, frame)
                images.append(frame)
        cap.release()
        return images
