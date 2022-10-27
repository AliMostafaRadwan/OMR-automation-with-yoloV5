import mss 
import cv2
import numpy as np



def start():
    
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        while True:
            screen = sct.grab(monitor)
            img_org = np.array(screen)
            img_crop =img_org[200:700, 290:1850]
            
            cv2.imshow('img', img_crop)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
start()