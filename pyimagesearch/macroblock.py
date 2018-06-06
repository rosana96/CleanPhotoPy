def getLuminance(pixel):
    R = pixel[2]
    G = pixel[1]
    B = pixel[0]
    y = (int)(0.299 * R + 0.587 * G + 0.114 * B)
    return y


class Macroblock:
    def __int__(self, image, x, y, dim):
        self.__image = image
        self.__x = x
        self.__y = y
        self.__dim = dim

    def meanSquaredError(self, blockA, blockB):
        (h1, w1) = blockA.shape[:2]
        (h2, w2) = blockB.shape[:2]
        mse = 0
        if h1 == h2 and w1 == w2:
            for i in range(0, w1):
                for j in range(0, h1):
                    pixelA = blockA[i, j]
                    pixelB = blockB[i, j]

                    yA = getLuminance(pixelA)
                    yB = getLuminance(pixelB)

                    mse += (yA - yB) ^ 2
            mse = mse / (w1 * h1)
            return mse
        else:
            pass
