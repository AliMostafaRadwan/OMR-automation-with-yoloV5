import cv2
import mss
import numpy as np
import keyboard
import time

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}


    while "Screen capturing":
        # Get raw pixels from the screen, save it to a Numpy array
        
        for i in range(400):
            screen = sct.grab(monitor)
            img_org = np.array(screen)
            h, w, channels = img_org.shape
            half2 = h//2
            
            bottom_whole= img_org[half2:, :]
            
            img4 = bottom_whole[230:390, 740:900]
            img3 = bottom_whole[230:390, 900:1050]
            img2 = bottom_whole[230:390, 1050:1200]
            img1 = bottom_whole[230:390, 1190:1380]
        
            time.sleep(2)
            cv2.imwrite(f'./imgTrain/{i}_img1_n.png',img1)
            cv2.imwrite(f'./imgTrain/{i}_img2_n.png',img2)
            cv2.imwrite(f'./imgTrain/{i}_img3_n.png',img3)
            cv2.imwrite(f'./imgTrain/{i}_img4_n.png',img4)
            keyboard.press_and_release('enter')
            
            print(i ,end='\r', flush=True)
        if i ==399:
            print('done')
            break