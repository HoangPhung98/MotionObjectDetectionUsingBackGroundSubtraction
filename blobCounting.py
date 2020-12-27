class BlobCounting:
    def __init__(self, verticalAxis, horizontalAxis):
        self.totalCount = 0
        self.horizontalAxis = horizontalAxis
        self.verticalAxis = verticalAxis
        print(self.verticalAxis," *****", self.horizontalAxis)

    def countVertical(self, blobs, countingRegion_minx, countingRegion_miny, countingRegion_maxx, countingRegion_maxy):
        count = 0
        for blob in blobs:
            if blob.miny < self.verticalAxis and blob.miny > countingRegion_miny and not blob.isCounted:
                count += 1
                blob.isCounted = True

        self.totalCount += count
        return self.totalCount
