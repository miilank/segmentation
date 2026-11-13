import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('data/images/image_10.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

# opseg
lower_yellow = np.array([20, 190, 207])
upper_yellow = np.array([35, 255, 255])

# binarna maska
mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
# nevalidni objekti
mask[0:110, 0:110] = 0
mask[200:270, 60:170] = 0

plt.figure()
plt.imshow(mask, cmap='gray')
plt.title("Mask (yellow)") 

# primjena maske
img_green_only = cv.bitwise_and(img, img, mask=mask)

plt.figure()
plt.imshow(img_green_only)
plt.title("Isolated yellow color")


plt.show()