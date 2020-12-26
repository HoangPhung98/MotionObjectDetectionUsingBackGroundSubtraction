import numpy as np

class Blob:
    color = (0, 255, 0)
    thickness = 1
    euclidDistanceThreshold = 4
    noiseBlobAreaThreshold = 110

    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

    def updateBoundary(self, x, y):
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x)
        self.maxy = max(self.maxy, y)

    def calEuclidDistance(self, x1, y1, x2, y2):
        euclid_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return euclid_distance

    def isNear(self, x1, y1, x, y):
        if self.calEuclidDistance(x1,y1,x,y) < self.euclidDistanceThreshold:
            return True
        else:
            return False

    def isNear2(self, x1, y1, x2, y2, x, y):
        # euclidDistance1 = self.calEuclidDistance(x1, y1, x, y)
        # euclidDistance2 = self.calEuclidDistance(x2, y2, x, y)
        # medianPosX = (x1+x2)//2
        # medianPosY = (y1+y2)//2
        # euclidDistance3 = self.calEuclidDistance(medianPosX, medianPosY, x, y)
        #euclidDistance = min(euclidDistance1, euclidDistance2, euclidDistance3)


        #tinh kc su dung tinh: kc tu 1 diem (x, y) den 1 duong thang di qua (x1,y1) (x2, y2)
        #ptdt co dang: ax + by + c = 0

        a = y1 - y2
        b = x2 - x1
        c = -a*x1 -b*y1

        distance = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

        if distance < self.euclidDistanceThreshold:
            return True
        else:
            return False

    def isBelongToThisBlob(self, x, y):

        if x >= self.minx and x <= self.maxx and y >= self.miny and y <= self.maxy:
            return True

        if x < self.minx:
            if y < self.miny:
                return self.isNear(self.minx, self.miny, x, y)
            else:
                if y < self.maxy:
                    return self.isNear2(self.minx, self.miny, self.minx, self.maxy, x, y)
                else:
                    return self.isNear(self.minx, self.maxy, x, y)
        else:
            if x < self.maxx:
                if y < self.miny:
                    return self.isNear2(self.minx, self.miny, self.maxx, self.miny, x, y)
                else:
                    if y > self.maxy:
                        return self.isNear2(self.minx, self.maxy, self.maxx, self.maxy, x, y)
            else:
                if y < self.miny:
                    return self.isNear(self.maxx, self.miny, x, y)
                else:
                    if y < self.maxy:
                        return self.isNear2(self.maxx, self.miny, self.maxx, self.maxy, x, y)
                    else:
                        return self.isNear(self.maxx, self.maxy, x, y)

        centerx = (self.minx + self.maxx) // 2
        centery = (self.miny + self.maxy) // 2

        # euclid distance
        educlid_distance = np.sqrt((centerx - x)**2 + (centery - y)**2)
        if educlid_distance <= self.euclidDistanceThreshold:
            return True
        else:
            return False

    def getArea(self):
        area = (self.maxx - self.minx)*(self.maxy - self.miny)
        return area

    def isThisBlobNoise(self):
        if self.getArea() < self.noiseBlobAreaThreshold:
            return True
        else:
            return False



