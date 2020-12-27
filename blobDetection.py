from blob import Blob
import numpy as np
class BlobDetetction:

    def __init__(self):
        self.totalNumberOfBlob = 0
        self.blobs = []
        self.prevBlobs = []


    def blobDetection(self, foreground, w, h, region_minx, region_miny, region_maxx, region_maxy):

        self.blobs.clear()
        for i in range(h):
            j = 0
            while j < w:
                if foreground[i][j] == 255:
                    if not self.belongToAlreadyExistedBlob(j, i, self.blobs):
                        # creat new blob that bound the current pixel
                        self.blobs.append(Blob(str(len(self.blobs)), False, j, i, j, i))
                j += 4

        self.filterNoiseBlobs(self.blobs)
        self.indexBlobs(self.blobs)
        self.mapBlobLabel()
        return self.blobs

    def mapBlobLabel(self):

        if len(self.prevBlobs) == 0:
            self.totalNumberOfBlob = len(self.blobs)
            self.prevBlobs = self.blobs
            return
        else:
            for x in self.prevBlobs:
                print(x.label, end=" ")
            print()
            for x in self.blobs:
                print(x.label, end=" ")
            print()
            for i in range(len(self.blobs)):
                for j in range(len(self.prevBlobs)):
                    if not self.prevBlobs[j].isLabelled:
                            if self.blobs[i].isMapOtherBlob(self.prevBlobs[j]):
                                self.blobs[i].label = self.prevBlobs[j].label
                                self.blobs[i].isCounted = self.prevBlobs[j].isCounted
                                self.blobs[i].isLabelled = True
                                self.prevBlobs[j].isLabelled = True
                                print(self.blobs[i].label,"-",self.prevBlobs[j].label)
                                continue

            for i in range(len(self.blobs)):
                if not self.blobs[i].isLabelled:
                    self.totalNumberOfBlob += 1
                    self.blobs[i].label = str(self.totalNumberOfBlob)
                    print(i,"--",self.totalNumberOfBlob)

            for x in self.prevBlobs:
                print(x.label, end=" ")
            print()
            for x in self.blobs:
                print(x.label, end=" ")
            print("***********")

            self.prevBlobs = np.copy(self.blobs)
            for i in range(len(self.prevBlobs)):
                self.prevBlobs[i].isLabelled = False

    def belongToAlreadyExistedBlob(self, x, y, blobs):

        for i in range(len(blobs)):
            blob = blobs[i]
            if blob.isBelongToThisBlob(x, y):
                blob.updateBoundary(x, y)
                return True

        return False

    def filterNoiseBlobs(self, blobs):
        lenBlobs = len(blobs)
        i = 0
        while i < lenBlobs:
            blob = blobs[i]
            if blob.isThisBlobNoise():
                blobs.remove(blob)
                lenBlobs -= 1
            else:
                i += 1

    def indexBlobs(self, blobs):
        for i in range(len(blobs)):
            blobs[i].label = str(i)


