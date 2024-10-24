import os
import cv2 as cv
import numpy as np

for root, path, f in os.walk("./"):
    for image in f:
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        equ = cv.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
        cv.imwrite(image + "_normalized.png" , res)
