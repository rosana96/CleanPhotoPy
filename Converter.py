class Converter:
    def bgrToYuv(self, pixelBGR):
        B = pixelBGR[0]
        G = pixelBGR[1]
        R = pixelBGR[2]

        Y = int(0.299 * R + 0.587 * G + 0.114 * B)
        U = int(-0.147 * R - 0.289 * G + 0.436 * B)
        V = int(0.615 * R - 0.515 * G - 0.100 * B)

        return [Y, U, V]

    def yuvToBgr(self, pixelYUV):
        Y = pixelYUV[0]
        U = pixelYUV[1]
        V = pixelYUV[2]

        R = int(Y + 1.140 * V)
        G = int(Y - 0.395 * U - 0.581 * V)
        B = int(Y + 2.032 * U)

        return [B, G, R]
