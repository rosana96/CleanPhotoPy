import numpy as np
import cv2


class ImageProcessor:
    def __init__(self, images, homographies, dim=8):
        self._images = images
        self._homographies = homographies
        (self._height, self._width) = images[0].shape[:2]
        self._dim = dim
        self._nrImg = len(images)

    def computeCleanImage(self):
        maxHeight = max(self._images, key=lambda im: im.shape[0])

        cleanImage = np.zeros((maxHeight, self._width * self._nrImg, 3), dtype="uint8")

        for i in range(0, self._height, self._dim):
            for j in range(0, self._width, self._dim):
                # calculam macroblockul(i, j) din imaginea finala
                INF = 1000000000
                minMeanSquaredError = INF
                idMinDiffImgPair = -1

                for k in range(0, self._nrImg - 1):
                    n = 0
                    MSE = 0
                    for y in range(i, min(i + self._dim, self._height)):
                        for x in range(j, min(j + self._dim, self._width)):
                            n += 1
                            MSE += meanSquareError(y, x, self._images[k], self._images.get[k + 1],
                                                   self._homographies[k])

                MSE /= n
                if MSE < minMeanSquaredError:
                    minMeanSquaredError = MSE
                    idMinDiffImgPair = k
                    #
                for y in range (i,min(i + self._dim, self._height)):
                    for x in range(j, min(j + self._dim, self._width)):
                        firstImgPixel = self._images[idMinDiffImgPair][y][x]

                        pts=[[x,y]]
                        pts = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))  # todo what does reshape do?
                        dst = cv2.perspectiveTransform(pts, self._homographies[idMinDiffImgPair])
                        [xp,yp]=dst[0][0]
                        [xp,yp]=[int(xp),int(yp)]
                        secondImgPixel = self._images[idMinDiffImgPair+1][xp][yp]

                        meanPixel = getMeanPixel(firstImgPixel, secondImgPixel)

                        cleanImage[x][y] = meanPixel



def getMeanPixel(firstImgPixel, secondImgPixel):
    pass



def meanSquareError(y, x, img1, img2, hom):
    pass


def getLuminance(pixel):
    R = pixel[2]
    G = pixel[1]
    B = pixel[0]
    y = (int)(0.299 * R + 0.587 * G + 0.114 * B)
    return y
