import cv2 as cv
import cv2.aruco as aruco
import numpy as np

# Function to rotate image
def rotate(img, angle):
    (height, width) = img.shape[:2]
    rotPoint = (width//2,height//2)
    # Get rotation matrix 
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)
    return cv.warpAffine(img, rotMat, dimensions)

# Place aruco marker on original image
def augmentImage(box, img, imgAug):
    h, w = imgAug.shape[:2]
    pts1 = box
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv.findHomography(pts2, pts1)  

     # It place aruco perfectly on desired square but surrounding is black 
    imgOut = cv.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # It darken the desired square
    cv.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    img = img + imgOut
    return img

# Function to find aruco marker in image
def findArucoMarkers(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{5}X{5}_{250}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(img,arucoDict,parameters=arucoParam)

    return corners[0][0]
def cropArucoImg(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 155, 255, cv.THRESH_BINARY)

    cont, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in cont: 
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.1 * peri, True)
        area = cv.contourArea(cnt)
        
        if len(approx) == 4: 
            if area > 100000 and area < 300000:
                rect = cv.minAreaRect(cnt)
                rotated_img = rotate(img, rect[2])
                rect = list(rect)
                box = cv.boxPoints(rect)
                box = np.int0(box)
    corner = findArucoMarkers(rotated_img)

    corner = np.int0(corner)
    a, b= np.sort([corner[0][1], corner[2][1]])
    c, d = np.sort([corner[0][0], corner[2][0]]) 

    cropped_img = rotated_img[a: b, c: d]
    return cropped_img


def placeAruco(img, lowerBound, upperBound, imgAug): 
    imgAug = cropArucoImg(imgAug)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lowerBound, upperBound)
    cont, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in cont: 
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.1 * peri, True)
        
        if len(approx) == 4 and cv.contourArea(cnt) > 10000:
            rect = cv.minAreaRect(cnt)
            diff_in_dmension = rect[1][0] - rect[1][1]
            if np.absolute(diff_in_dmension) < 10: 
                box = cv.boxPoints(rect)
                box = np.float32(box)
                img = augmentImage(box, img, imgAug)
                return img


img = cv.imread('Photos/CVtask.jpg')

imgAug = cv.imread('Photos/LMAO.jpg')
lower_bound = np.array([30, 140, 190])
upper_bound = np.array([60, 170, 220])
img = placeAruco(img, lower_bound, upper_bound, imgAug)

imgAug = cv.imread('Photos/XD.jpg')
lower_bound = np.array([15, 200, 20])
upper_bound = np.array([30, 255, 255])
img = placeAruco(img, lower_bound, upper_bound, imgAug)

imgAug = cv.imread('Photos/Ha.jpg')
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([2, 2, 2])
img = placeAruco(img, lower_bound, upper_bound, imgAug)

imgAug = cv.imread('Photos/HaHa.jpg')
lower_bound = np.array([15, 15, 225])
upper_bound = np.array([25, 25, 235])
img = placeAruco(img, lower_bound, upper_bound, imgAug)

cv.imshow('Final Image',img)

cv.waitKey(0)
cv.destroyAllWindows()