class BlobCounting:
    countingWidth = 50
    def __init__(self, verticalAxis, horizontalAxis):
        self.totalCount = 0
        self.horizontalAxis = horizontalAxis
        self.verticalAxis = verticalAxis
        print(self.verticalAxis," *****", self.horizontalAxis)

    def countVertical(self, blobs):
        count = 0
        for blob in blobs:
            if blob.miny < self.verticalAxis and blob.miny > self.verticalAxis - self.countingWidth and not blob.isCounted:
                count += 1
                print("count:", count)
                blob.isCounted = True

        self.totalCount += count
        print("totalCount", self.totalCount)
        return self.totalCount
