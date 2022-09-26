
import torch
import numpy as np
import cv2
import mss
import win32api
from green_cells import green_cell
import threading
import keyboard
import time
import multiprocessing
import pyautogui
import os
class ObjectDetection:
    
    def __init__(self):

        # run it on gpu if available
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)



    def load_model(self):

        return torch.hub.load('ultralytics/yolov5', 'custom', path='one_at_time.pt',force_reload=True)


    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.4:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                bgr2 = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                cv2.putText(frame, f'{row[4]:.2f}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr2, 2)
                
        return frame
    global locker
    locker = threading.Lock()
    def __call__(self):
        threading.Thread(target=self.main).start()

    def main(self):
        
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            while "Screen capturing":
                screen = sct.grab(monitor)
                img_org = np.array(screen)
                h, w, channels = img_org.shape
                half2 = h//2
                bottom1 = img_org[half2:, :]
                bottom2 = img_org[half2:, :]
                bottom3= img_org[half2:, :]
                bottom4 = img_org[half2:, :]
                
                img_crp = img_org[200:700, 290:1850]
                img4 = bottom4[230:390, 720:880]
                img3 = bottom3[230:390, 880:1070]
                img2 = bottom2[230:390, 1050:1250]
                img1 = bottom1[230:390, 1250:1450]

                results1 = self.score_frame(img1)
                results2 = self.score_frame(img2)
                results3 = self.score_frame(img3)
                results4 = self.score_frame(img4)

                # img = self.plot_boxes(results1, img)
                img1 = self.plot_boxes(results1, img1)
                img2 = self.plot_boxes(results2, img2)
                img3 = self.plot_boxes(results3, img3)
                img4 = self.plot_boxes(results4, img4)
                x,y = green_cell()
                def typing_answer():
                    try:
                        locker.acquire()
                        if self.classes[int(results1[0][0])] == 'correct':
                            print("A")
                            time.sleep(1.5)
                            keyboard.press_and_release('a')
                            time.sleep(0.4)
                            keyboard.press_and_release('enter')
                            time.sleep(1.5)
                        elif self.classes[int(results2[0][0])] == 'correct':
                            print("B")
                            time.sleep(1.5)
                            keyboard.press_and_release('b')
                            time.sleep(0.4)
                            keyboard.press_and_release('enter')
                            time.sleep(1.5) 
                        elif self.classes[int(results3[0][0])] == 'correct':
                            print("C")
                            time.sleep(1.5)
                            keyboard.press_and_release('c')
                            time.sleep(0.4)
                            keyboard.press_and_release('enter')
                            time.sleep(1.5) 
                        elif self.classes[int(results4[0][0])] == 'correct':
                            print("D")
                            time.sleep(1.5)
                            keyboard.press_and_release('d')
                            time.sleep(0.4)
                            keyboard.press_and_release('enter')
                            time.sleep(1.5) 
                    except Exception:
                        print("Can't detect")
                    finally:
                        locker.release()
                
                threading.Thread(target=typing_answer).start()
                
                # cv2.imshow('img_croped', img_crp)
                cv2.imshow('img1', img1)
                cv2.imshow('img2', img2)
                cv2.imshow('img3', img3)
                cv2.imshow('img4', img4)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                if win32api.GetAsyncKeyState(0x1B):
                    # kill the terminal
                    os.system("taskkill /f /im cmd.exe")
                    break

# Create a new object and execute.
if __name__ == '__main__':
    locker.acquire()
    p1 = multiprocessing.Process(target=green_cell)
    p1.start()
    locker.release()
    detection = ObjectDetection()
    detection()
#