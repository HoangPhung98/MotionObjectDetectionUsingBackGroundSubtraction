from blob import Blob
import cv2 as cv
import  numpy as np

# img = [[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)],
#        [(0,0,0),(0,0,0),(255,255,255),(255,255,255),(0,0,0),(0,0,0)],
#        [(0,0,0),(0,0,0),(0,0,0),(255,255,255),(0,0,0),(0,0,0)],
#        [(0,0,0),(255,255,255),(0,0,0),(0,0,0),(0,0,0),(0,0,0)],
#        [(255,255,255),(255,255,255),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]]
# img = np.array(img)

def blobDetection(foreground):
    blobs = []
    w = len(foreground[0])
    h = len(foreground)
    for i in range(h):
        for j in range(w):
            if foreground[i][j][0] == 255:
                if not belongToAlreadyExistedBlob(j,i, blobs):
                    # creat new blob that bound the current pixel
                    blobs.append(Blob(j,i,j,i))

    filterNoiseBlobs(blobs)
    return blobs

def belongToAlreadyExistedBlob(x, y, blobs):

    for i in range(len(blobs)):
        blob = blobs[i]
        if blob.isBelongToThisBlob(x,y):
            blob.updateBoundary(x,y)
            return True

    return False

def filterNoiseBlobs(blobs):
    lenBlobs = len(blobs)
    i=0
    while i<lenBlobs:
        blob = blobs[i]
        if blob.isThisBlobNoise():
            blobs.remove(blob)
            lenBlobs-=1
        else:
            i+=1


def frameBlobs(blobs, img):
    for i in range(len(blobs)):
        blob = blobs[i]
        cv.rectangle(img, (blob.minx-1, blob.miny-1), (blob.maxx+1, blob.maxy+1), blob.color, blob.thickness)

img = cv.imread("testMask.jpg")

blobs = blobDetection(img)
frameBlobs(blobs, img)
img = img.astype(np.uint8)
cv.imwrite("out_testMask.jpg", img)
cv.imshow("blobs", img)
cv.waitKey()
