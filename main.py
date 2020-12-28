import cv2 as cv
import numpy as np
from blob import Blob
from blobDetection import BlobDetetction
from blobCounting import BlobCounting

totalNumberOfBlob = 0
realNumberOfCar = 16
numBlob=0


def drawBlobCounting(numBlob, img, verticalAxis, horizontalAxis, countingWidth, w, h):
    if(verticalAxis!=0):
        cv.line(img, (0, verticalAxis), (w, verticalAxis), (200, 180, 0), 3)
        cv.line(img, (0, verticalAxis-countingWidth), (w, verticalAxis-countingWidth), (200, 180, 0), 3)

    else:
        cv.line(img, (horizontalAxis, 0), (horizontalAxis, h), (255,0,0), 3)
    cv.putText(img,
               "passed:",
               (20, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.8,
               color=(0, 255, 255),
               lineType=3,
               thickness=1)
    cv.putText(img,
               str(numBlob),
               (120, 55),
               fontFace=cv.FONT_HERSHEY_PLAIN,
               fontScale=3,
               color=(0, 255, 255),
               lineType=3,
               thickness=2)


def drawCurrentNumberOfBlob(blobs, img):
    cv.putText(img,
               "current:",
               (20, 100),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.8,
               color=(0, 255, 255),
               lineType=3,
               thickness=1)
    cv.putText(img,
               str(len(blobs)),
               (120, 105),
               fontFace=cv.FONT_HERSHEY_PLAIN,
               fontScale=3,
               color=(0, 255, 255),
               lineType=3,
               thickness=2)


def frameDiff(uri):

    video = cv.VideoCapture(uri)
    ret, pre_img = video.read()
    shape = pre_img.shape
    pre_img = cv.cvtColor(pre_img, cv.COLOR_BGR2GRAY)
    pre_img = pyrImg(pre_img, shape[1], shape[0])

    shape = pre_img.shape
    h = shape[0]
    w = shape[1]

    pre_sub = pre_img*0

    # window
    cv.namedWindow("original")
    cv.resizeWindow("original", w, h)
    cv.namedWindow("mask_diff")
    cv.resizeWindow("mask_diff", w, h)

    #write video
    out_fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_diff = cv.VideoWriter("out_diff.avi", out_fourcc, 25.0, (len(pre_img[0]), len(pre_img)), True)

    # previous blobs
    blobDetection = BlobDetetction()
    blobCounting = BlobCounting(4 * h // 5, 4 * w // 5)

    # detecting region
    detectingRegion_minx = 0
    detectingRegion_miny = h // 2
    detectingRegion_maxx = w
    detectingRegion_maxy = h

    # couting region
    # countingRegion_minx = 0
    # countingRegion_miny = blobCounting.verticalAxis - 50
    # countingRegion_maxx = w
    # countingRegion_maxy = h
    f = open("diff.txt","a")


    while True:
        # recent frame
        ret, img = video.read()
        if ret == False:
            f.write(str(numBlob))
            f.write("\n")
            f.write(str(realNumberOfCar / numBlob))
            f.close()
        img = pyrImg(img, w, h)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # sub 2 frame
        sub = cv.absdiff(img_gray, pre_img)

        diff = cv.absdiff(sub, pre_sub)
        # update pre_img
        pre_img = img_gray
        pre_sub = sub
        # threshode
        ret, mask = cv.threshold(diff, 15, 255, cv.THRESH_BINARY)
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv.dilate(mask, kernel)
        # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,kernel)
        blobs = blobDetection.blobDetection(mask, w, h, detectingRegion_minx, detectingRegion_miny, detectingRegion_maxx, detectingRegion_maxy)
        frameBlobs(blobs, img)

        numBlob = blobCounting.countVertical(blobDetection.prevBlobs)
        drawBlobCounting(numBlob, img, blobCounting.verticalAxis, blobCounting.horizontalAxis, blobCounting.countingWidth, w, h)
        drawCurrentNumberOfBlob(blobs, img)

        # write video
        out_diff.write(img)

        cv.imshow("original", img)
        cv.imshow("mask_diff", mask)

        key = cv.waitKey(1)
        if key == ord('q'):
            break;
        if key == ord(' '):
            cv.waitKey()

    cv.destroyAllWindows()


def mean(n, uri):

    video = cv.VideoCapture(uri)
    ret, img = video.read()



    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = img_gray.shape
    img_gray = pyrImg(img_gray, shape[1], shape[0])

    shape = img_gray.shape
    h = shape[0]
    w = shape[1]

    cv.namedWindow("original")
    cv.resizeWindow("original", w, h)
    cv.namedWindow("mask_mean")
    cv.resizeWindow("mask_mean", w, h)

    nframe = []
    sum = np.zeros((h, w))
    for i in range(n):
        ret, img = video.read()
        img = pyrImg(img,w,h)

        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        nframe.append(img_gray)
        sum = sum + img_gray
    mean = np.uint8(sum // n)

    pos = 0

    blobDetection = BlobDetetction()
    blobCounting = BlobCounting(4 * h // 5, 4 * w // 5)
    # detecting region
    detectingRegion_minx = 0
    detectingRegion_miny = h // 2
    detectingRegion_maxx = w
    detectingRegion_maxy = h

    f = open("mean.txt","a")

    out_fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_mean = cv.VideoWriter("out_mean.avi", out_fourcc, 25.0, (w, h), True)
    while True:
        ret, img = video.read()

        if ret == False:
            f.write(str(numBlob))
            f.write("\n")
            f.write(str(realNumberOfCar / numBlob))
            f.close()

        img = pyrImg(img,w,h)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        sub = cv.absdiff(img_gray, mean)
        ret, mask = cv.threshold(sub, 25, 255, cv.THRESH_BINARY)

        blobs = blobDetection.blobDetection(mask, w, h,
                                            detectingRegion_minx, detectingRegion_miny,
                                            detectingRegion_maxx, detectingRegion_maxy)
        frameBlobs(blobs, img)

        numBlob = blobCounting.countVertical(blobDetection.prevBlobs)
        drawBlobCounting(numBlob, img, blobCounting.verticalAxis, blobCounting.horizontalAxis,
                         blobCounting.countingWidth, w, h)
        drawCurrentNumberOfBlob(blobs, img)

        sum = (sum - nframe[pos] + img_gray)
        mean = np.uint8(sum // n)
        nframe[pos] = img_gray
        if pos < n - 1:
            pos = pos + 1
        else:
            pos = 0

        # write video
        out_mean.write(img)

        cv.imshow("original", img)
        cv.imshow("mask_mean", mask)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            cv.waitKey()

    cv.destroyAllWindows()

def mog(uri):

    video = cv.VideoCapture(uri)
    ret, img = video.read()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = img_gray.shape
    img_gray = pyrImg(img_gray, shape[1], shape[0])

    shape = img_gray.shape
    h = shape[0]
    w = shape[1]

    cv.namedWindow("original")
    cv.resizeWindow("original", w, h)
    cv.namedWindow("mask_mog")
    cv.resizeWindow("mask_mog", w, h)

    backsub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


    blobDetection = BlobDetetction()
    blobCounting = BlobCounting(4 * h // 5, 4 * w // 5)
    # detecting region
    detectingRegion_minx = 0
    detectingRegion_miny = h // 2
    detectingRegion_maxx = w
    detectingRegion_maxy = h

    f = open("mog.txt","a")

    out_fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_mog = cv.VideoWriter("out_mog.avi", out_fourcc, 25.0, (w, h), True)

    while True:
        # recent frame
        ret, img = video.read()

        if ret == False:
            f.write(str(numBlob))
            f.write("\n")
            f.write(str(realNumberOfCar / numBlob))
            f.close()

        img = pyrImg(img,w,h)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        mask = backsub.apply(img_gray)
        kernel = np.ones((3, 3), np.uint8)
        # mask = cv.dilate(mask, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        blobs = blobDetection.blobDetection(mask, w, h,
                                            detectingRegion_minx, detectingRegion_miny,
                                            detectingRegion_maxx, detectingRegion_maxy)
        frameBlobs(blobs, img)

        numBlob = blobCounting.countVertical(blobDetection.prevBlobs)
        drawBlobCounting(numBlob, img, blobCounting.verticalAxis, blobCounting.horizontalAxis,
                         blobCounting.countingWidth, w, h)
        drawCurrentNumberOfBlob(blobs, img)

        # write video
        out_mog.write(img)

        cv.imshow("original", img)
        cv.imshow("mask_mog", mask)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            cv.waitKey()
    f.write(blobDetection.totalNumberOfBlob)
    f.close()

def all(n, uri):
    video = cv.VideoCapture(uri)
    ret, img = video.read()
    shape = img.shape
    h = shape[0]
    w = shape[1]
    w = w // 2
    h = h // 2
    # mean
    nframe = []
    sum = np.zeros((h, w))
    for i in range(n):
        ret, img = video.read()
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.resize(img, (w, h))

        nframe.append(img)
        sum = sum + img
    mean = np.uint8(sum // n)
    pos = 0

    # diff
    pre_img = img
    pre_sub = pre_img

    # mog
    backsub = cv.createBackgroundSubtractorMOG2()

    while True:
        ret, img = video.read()
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.resize(img, (w, h))

        # mean
        sub_mean = cv.absdiff(img, mean)
        ret, mask_mean = cv.threshold(sub_mean, 40, 255, cv.THRESH_BINARY)

        sum = (sum - nframe[pos] + img)
        mean = np.uint8(sum // n)
        nframe[pos] = img
        if pos < n - 1:
            pos = pos + 1
        else:
            pos = 0

        # diff
        # sub 2 frame
        sub_diff = cv.absdiff(img, pre_img)

        diff = cv.absdiff(sub_diff, pre_sub)
        # update pre_img
        pre_img = img
        pre_sub = sub_diff

        # threshode
        ret, mask_diff = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

        # mog
        mask_mog = backsub.apply(img)
        mask_mog = cv.morphologyEx(mask_mog, cv.MORPH_OPEN, (5, 5))

        cv.namedWindow("original")
        cv.resizeWindow("original", w, h)
        cv.namedWindow("mask_diff")
        cv.resizeWindow("mask_diff", w, h)
        cv.namedWindow("mask_mean")
        cv.resizeWindow("mask_mean", w, h)
        cv.namedWindow("mask_mog")
        cv.resizeWindow("mask_mog", w, h)
        cv.imshow("mask_diff", mask_diff)
        cv.imshow("mask_mean", mask_mean)
        cv.imshow("mask_mog", mask_mog)
        cv.imshow("original", img)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            cv.waitKey()

def frameBlobs(blobs, img):
    for i in range(len(blobs)):
        blob = blobs[i]
        cv.rectangle(img, (blob.minx-1, blob.miny-1), (blob.maxx+1, blob.maxy+1), blob.color, blob.thickness)
        # cv.putText(img,
        #            blob.label,
        #            (blob.minx, blob.miny),
        #            fontFace=cv.FONT_HERSHEY_PLAIN,
        #            fontScale=2,
        #            color=(0,255,255),
        #            lineType=3,
        #            thickness=2)

def pyrImg(img, w, h):
    return cv.pyrDown(img)

# main
uri_road = r"road.mp4"
uri_road_1D = r"road_1D.mp4"
uri_human = r"human.mp4"
uri_moto_vietnam = r"moto_vietnam_SD.mp4"
uri_topdown = r"topdownview.mp4"
uri_topdown2 = r"topdownview2.mp4"

uri_trafficgood = r"traffic_good.mp4"
uri_trafficgood2 = r"traffic_good_2.mp4"

uri_trafficgood3 = r"traffic_good_3.mp4"
uri_trafficgood3cut = r"traffic_good_3_cut.mp4"


#
# frameDiff(uri_road)
# frameDiff(uri_road_1D)
# frameDiff(uri_human)
# frameDiff(uri_topdown2)
# frameDiff(uri_trafficgood)

frameDiff(uri_trafficgood3cut)

# mean(3, uri_trafficgood3cut)

# mog(uri_trafficgood3cut)

