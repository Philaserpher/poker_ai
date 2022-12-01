import cv2
from PIL import ImageGrab
import numpy as np


colour = None


# ////////////////////////////////////////////////////////////////// GRAB IMAGE FROM DESKTOP SCREEN //////////////////////////////////////////////////////////////////

image = ImageGrab.grab(
    bbox=None, include_layered_windows=False, all_screens=True).convert("RGB")
image = np.array(image)
image = image[:, :, ::-1].copy()
cropped_image = image[452:704, 2760:2962]


# ////////////////////////////////////////////////////////////////// DETECT CARD COLOUR //////////////////////////////////////////////////////////////////

temp_img = cropped_image.copy()
# Using RGB instead of BGR here to flip and look for blue
# Unsure how effective it is, I read that blue is easier to detect
temp_hsv = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)

# define range of blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(temp_hsv, lower_blue, upper_blue)
#masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
#cv2.imshow('image', masked_image)
# cv2.waitKey(0)
count = mask > 250
count = np.count_nonzero(count)
colour = "red" if count > 20 else "black"
print(colour)


# ////////////////////////////////////////////////////////////////// CONVERT IMAGE TO BLACK AND WHITE //////////////////////////////////////////////////////////////////

cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

(thresh, cropped_image) = cv2.threshold(
    cropped_image, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# ////////////////////////////////////////////////////////////////// SHOW IMAGE //////////////////////////////////////////////////////////////////

cv2.imshow('image', cropped_image)
cv2.waitKey(0)
