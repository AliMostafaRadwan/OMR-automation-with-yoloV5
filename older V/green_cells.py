import mss
import cv2
import numpy as np
import pyautogui
def green_cell():
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        while "Screen capturing":
            screen = sct.grab(monitor)
            img = np.array(screen)
            img_crp = img[200:700, 325:1860]
            hsv = cv2.cvtColor(img_crp, cv2.COLOR_BGR2HSV)
            lower_green = np.array([10, 120, 0])
            upper_green = np.array([85, 255, 255])
            global mask
            mask = cv2.inRange(hsv, lower_green, upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 15 and h > 15:
                    # print(x, y)
                    x = x+330
                    y = y+210
                    break
            return x, y
        
green_cell()

def IS_EMPTY():
    if cv2.countNonZero(mask) <= 500:
        return True
    elif cv2.countNonZero(mask) > 500:
        return False