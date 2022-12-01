import cv2
from PIL import ImageGrab
import numpy as np


colour = None


# ////////////////////////////////////////////////////////////////// GRAB IMAGE FROM DESKTOP SCREEN //////////////////////////////////////////////////////////////////

def grab_image():
    img = ImageGrab.grab(
        bbox=None, include_layered_windows=False, all_screens=True).convert("RGB")
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    img = img[452:704, 2760:2962]
    return img


# ////////////////////////////////////////////////////////////////// DETECT CARD COLOUR //////////////////////////////////////////////////////////////////

def detect_colour(img):
    temp_img = img.copy()
    # Using RGB instead of BGR here to flip and look for blue
    # Unsure how effective it is, I read that blue is easier to detect
    temp_hsv = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(temp_hsv, lower_blue, upper_blue)
    # masked_image = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('image', masked_image)
    # cv2.waitKey(0)
    count = mask > 250
    count = np.count_nonzero(count)
    colour = "red" if count > 20 else "black"
    return colour


# ////////////////////////////////////////////////////////////////// CONVERT IMAGE TO BLACK AND WHITE //////////////////////////////////////////////////////////////////

def c2bw(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, img) = cv2.threshold(
        img, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img

# ////////////////////////////////////////////////////////////////// SHOW IMAGE //////////////////////////////////////////////////////////////////


def show_image(img, colour):
    cv2.imshow(f'{colour.capitalize()} card', img)
    cv2.waitKey(0)


# ////////////////////////////////////////////////////////////////// RUN (DEBUG ONLY) //////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    img = grab_image()
    colour = detect_colour(img)
    img_bw = c2bw(img)
    show_image(img_bw, colour)
