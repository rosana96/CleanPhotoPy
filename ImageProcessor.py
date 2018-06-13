import abc

import cv2
import numpy as np

from utils import Matcher
from utils.Converter import Converter


class AbstractImageProcessor:
    __metaclass__ = abc.ABCMeta

    def __init__(self, images, dim=20):
        self._images = images
        (self._height, self._width) = images[0].shape[:2]
        self._dim = dim
        self._nrImg = len(images)
        self._imRef = images[(len(images)-1) // 2]

    def reconstructCleanImage(self):

        cleanImage = np.zeros((self._height, self._width, 3), dtype="uint8")
        # cleanImage = np.copy(self._imRef)
        for i in range(0, self._height, self._dim):
            for j in range(0, self._width, self._dim):
                # calculam macroblockul(i, j) din imaginea finala
                INF = 10000000000000
                minMeanSquaredError = INF
                idMinDiffImgPair = -1
                delta = 1

                for k in range(0, self._nrImg - 1):
                    delta = 1
                    for step in range(1, 4):
                        if k + step < self._nrImg:
                            try:
                                n = 0
                                MSE = 0
                                for y in range(i, min(i + self._dim, self._height)):
                                    for x in range(j, min(j + self._dim, self._width)):
                                        n += 1
                                        MSE += self.meanSquareError(y, x, k, step)

                                MSE /= n
                                if MSE < minMeanSquaredError:
                                    minMeanSquaredError = MSE
                                    idMinDiffImgPair = k
                                    delta = step
                            except Exception:
                                pass

                if idMinDiffImgPair == -1:
                    continue

                self.reconstructBlock(i, j, cleanImage, idMinDiffImgPair, delta)

        return cleanImage

    def getLuminance(self, pixel):
        R = pixel[2]
        G = pixel[1]
        B = pixel[0]
        y = int(0.299 * R + 0.587 * G + 0.114 * B)
        return y

    def getMeanPixel(self, firstImgPixel, secondImgPixel):
        converter = Converter()
        firstYUV = converter.bgrToYuv(firstImgPixel)
        secondYUV = converter.bgrToYuv(secondImgPixel)

        Y = (firstYUV[0] + secondYUV[0]) // 2
        U = (firstYUV[1] + secondYUV[1]) // 2
        V = (firstYUV[2] + secondYUV[2]) // 2

        yuvResult = [Y, U, V]
        return converter.yuvToBgr(yuvResult)

    @abc.abstractmethod
    def meanSquareError(self, y, x, k, delta):
        return

    @abc.abstractmethod
    def reconstructBlock(self, i, j, cleanImage, idMinDiffImgPair, delta):
        return


class MovingCameraImageProcessor(AbstractImageProcessor):
    def __init__(self, images, dim=20):
        super().__init__(images, dim)
        self._HREF = self.calculateHomographies()

    def calculateHomographies(self):
        matcher = Matcher.Matcher()
        Href = []
        for i in self._images:
            H = matcher.getHomography([i, self._imRef])
            Href.append(H)
        return Href

    def reconstructBlock(self, i, j, cleanImage, idMinDiffImgPair, delta=1):
        for y in range(i, min(i + self._dim, self._height)):
            for x in range(j, min(j + self._dim, self._width)):
                print("id min diff -------------  " + str(i) + " " + str(j)+ "STEP: " + str(delta))
                print(idMinDiffImgPair)

                pts = [[x, y]]
                pts = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))

                dst = cv2.perspectiveTransform(pts, self._HREF[idMinDiffImgPair])
                [xp1, yp1] = dst[0][0]
                [xp1, yp1] = [int(xp1), int(yp1)]

                dst = cv2.perspectiveTransform(pts, self._HREF[idMinDiffImgPair + delta])
                [xp2, yp2] = dst[0][0]
                [xp2, yp2] = [int(xp2), int(yp2)]

                xp1 = min(max(xp1, 0), self._width - 1)
                xp2 = min(max(xp2, 0), self._width - 1)
                yp1 = min(max(yp1, 0), self._height - 1)
                yp2 = min(max(yp2, 0), self._height - 1)

                pixel = self._images[idMinDiffImgPair][yp1][xp1]
                # pixel = self.getMeanPixel(self._images[idMinDiffImgPair][yp1][xp1], self._images[idMinDiffImgPair+delta][yp2][xp2])
                cleanImage[y][x] = pixel

    def meanSquareError(self, y, x, imgPos, delta):
        img1 = self._images[imgPos]
        img2 = self._images[imgPos + delta]
        Href1 = self._HREF[imgPos]
        Href2 = self._HREF[imgPos + delta]

        pts = np.array([[x, y]], dtype=np.float32).reshape((-1, 1, 2))
        trans1 = cv2.perspectiveTransform(pts, Href1)[0][0]
        trans2 = cv2.perspectiveTransform(pts, Href2)[0][0]

        # print(trans1)
        #
        # print(trans2)
        # print("=======================")
        # print([x, y])
        # print("***********************")

        xp1 = int(trans1[0])
        yp1 = int(trans1[1])
        xp2 = int(trans2[0])
        yp2 = int(trans2[1])

        # small errors are accepted
        err = 5
        if xp1 not in range(-err, self._width + err):
            raise Exception
        if xp2 not in range(-err, self._width + err):
            raise Exception
        if yp1 not in range(-err, self._height + err):
            raise Exception
        if yp2 not in range(-err, self._height + err):
            raise Exception

        xp1 = min(max(xp1, 0), self._width - 1)
        xp2 = min(max(xp2, 0), self._width - 1)
        yp1 = min(max(yp1, 0), self._height - 1)
        yp2 = min(max(yp2, 0), self._height - 1)

        pixel1 = img1[yp1][xp1]
        pixel2 = img2[yp2][xp2]

        # cv2.imshow("Image A", img1)
        # cv2.imshow("Image B", img2)
        # cv2.imshow("ref", self._imRef)

        return (self.getLuminance(pixel1) - self.getLuminance(pixel2)) ** 2


class StillCameraImageProcessor(AbstractImageProcessor):
    def __init__(self, images, dim=64):
        super().__init__(images, dim)

    def reconstructBlock(self, i, j, cleanImage, idMinDiffImgPair, delta=1):
        for y in range(i, min(i + self._dim, self._height)):
            for x in range(j, min(j + self._dim, self._width)):
                print("STILL: id min diff -------------  " + str(y) + " " + str(x)+ "STEP: " + str(delta))
                print(idMinDiffImgPair)

                # pixel = self.getMeanPixel(self._images[idMinDiffImgPair][y][x], self._images[idMinDiffImgPair+delta][y][x])
                pixel = self._images[idMinDiffImgPair][y][x]
                cleanImage[y][x] = pixel

    def meanSquareError(self, y, x, imgPos, delta=1):
        img1 = self._images[imgPos]
        img2 = self._images[imgPos + delta]

        pixel1 = img1[y][x]
        pixel2 = img2[y][x]

        return (self.getLuminance(pixel1) - self.getLuminance(pixel2)) ** 2
