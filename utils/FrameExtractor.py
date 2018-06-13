import math


import cv2

class FrameExtractor:
    def __init__(self, videoFile):
        self._videoFile = videoFile
        self._frames = []

    def extractFrames(self):

        imagesFolder = "images/frames"
        images=[]
        #print(cv2.__version__)
        cap = cv2.VideoCapture(self._videoFile)
        cap.open(self._videoFile)
        frameRate = cap.get(5)  # frame rate
        while (cap.isOpened()):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if frameId % (math.floor(frameRate)//2) == 0:
                filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
                cv2.imwrite(filename, frame)
                images.append(frame)
        cap.release()
        return images

