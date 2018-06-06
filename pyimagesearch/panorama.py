# import the necessary packages
import cv2
import imutils
import numpy as np


def testHomography(matrix, imageA, imageB):
    blockDim = 100
    start = 0
    end = start + blockDim
    pts = []
    for x in range(start, end):
        for y in range(start, end):
            pts.append([x, y])

    pts = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))  # todo what does reshape do?
    dst = cv2.perspectiveTransform(pts, matrix)
    b = []
    minx = 10000
    maxx = 0
    miny = 10000
    maxy = 0
    for pair in dst:
        xp = int(pair[0][0])
        yp = int(pair[0][1])
        if xp > maxx:
            maxx = xp
        if xp < minx:
            minx = xp
        if yp > maxy:
            maxy = yp
        if yp < miny:
            miny = yp
        print(str(xp) + "   " + str(yp) + "\n")
        b.append([xp, yp])

    miny = max(miny, 0)
    minx = max(minx, 0)
    crop_img = imageA[miny:(miny + blockDim), minx:(minx + blockDim)]
    orig_img = imageB[start:end, start:end]
    cv2.imshow("orig", orig_img)
    cv2.imshow("cropped", crop_img)

    wp = cv2.warpPerspective(imageB, matrix, (700, 700))
    # cv2.imshow("wp", wp)
    # cv2.imshow("a", imageA)

    pass


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def getHomography(self, images, ratio=0.75, reprojThresh=4.0,  # todo what are these
                      showMatches=True):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageA, imageB) = images
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        (kpsA, featuresA) = self.detectAndDescribe(imageA)

        # match features between the two images
        M = self.matchKeypoints(kpsB, kpsA,
                                featuresB, featuresA, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        testHomography(H, imageA, imageB)

        return H

        result = cv2.warpPerspective(imageB, H,
                                     (imageB.shape[1] + imageA.shape[1], imageB.shape[0]+100))
        cv2.imshow("b", result)

        result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

        return H

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (keypoints, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            keypoints = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (keypoints, features) = extractor.compute(gray, keypoints)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        keypoints = np.float32([kp.pt for kp in keypoints])

        # return a tuple of keypoints and features
        return (keypoints, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageB, imageA, kpsB, kpsA, matches, status):
        # initialize the output visualization image
        (heightA, widthA) = imageA.shape[:2]
        (heightB, widthB) = imageB.shape[:2]
        vis = np.zeros((max(heightA, heightB), widthA + widthB, 3), dtype="uint8")
        vis[0:heightA, 0:widthA] = imageA
        vis[0:heightB, widthA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + widthA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis