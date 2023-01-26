
import torch
import numpy as np
import cv2
import mss
import threading
import keyboard
import time
import tkinter
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import sv_ttk
class ObjectDetection:
    
    def __init__(self):
        """
        The function loads the model, gets the classes, and sets the device to GPU if available
        """
        # run it on gpu if available
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.valid = False
        self.count = 0
        print("\n\nDevice Used:",self.device)


    """
    > Loads the model from the path specified in the function
    :return: The model is being returned.
    """

    def load_model(self):
        return torch.hub.load('model_repo\yolov5', model='custom', path='best.pt', source='local', force_reload=True)

    """
    > The function takes a single frame as input, and returns the labels and coordinates of the detected
    objects
    
    :param frame: a single image
    :return: The labels and the cordinates of the bounding boxes.
    """
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    """
    > This function takes in a class number and returns the corresponding class label
    
    :param x: the input data
    :return: The class of the image.
    """

    def class_to_label(self, x):
        return self.classes[int(x)]


    """
    > The function takes in the results of the model and the frame, and returns the frame with the boxes
    drawn on it
    
    :param results: a list of tuples, each tuple contains the label and the coordinates of the bounding
    box
    :param frame: The image to be processed
    :return: the frame with the boxes drawn on it.
    """
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
# A global variable that is used to lock the thread.
    global locker
    locker = threading.Lock()
    def __call__(self):
        threading.Thread(target=self.main).start()





    def main(self):  # sourcery skip: merge-nested-ifs
        ikkuna=tkinter.Tk()
        ikkuna.title("automation for RemarkOffice software with ML made by ali mostafa")
        ikkuna.geometry("800x400")
        ikkuna.resizable(0,0)
        sv_ttk.set_theme("dark")
        ####
        paneeli_image=tkinter.Label(ikkuna) #,image=img)
        paneeli_image.grid(row=0,column=2,padx=90)
        ####
        # paneeli_image2=tkinter.Label(ikkuna) #,image=img)
        # paneeli_image2.grid(row=0,column=1,columnspan=1,pady=1,padx=10)
        # ####
        # paneeli_image3=tkinter.Label(ikkuna) #,image=img)
        # paneeli_image3.grid(row=0,column=2,columnspan=1,pady=1,padx=10)
        # ####
        # paneeli_image4=tkinter.Label(ikkuna) #,image=img)
        # paneeli_image4.grid(row=0,column=3,columnspan=1,pady=1,padx=10)
        # ####




        my_lable = Label(ikkuna, text=" ",font=('Helvetica',22))
        my_lable.grid(row=1, column=2, padx=10, pady=10)

        def otakuva():
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
                    bottom_whole= img_org[half2:, :]
                    img4 = bottom4[230:390, 740:900]
                    img3 = bottom3[230:390, 900:1050]
                    img2 = bottom2[230:390, 1050:1200]
                    img1 = bottom1[230:390, 1190:1380]
                    img_whole = bottom_whole[230:390, 760:1380]

                    results1 = self.score_frame(img1)
                    results2 = self.score_frame(img2)
                    results3 = self.score_frame(img3)
                    results4 = self.score_frame(img4)
                    results_whole = self.score_frame(img_whole)

                    
                    # try:
                    #     if self.classes[int(results1[0][0])] == 'correct':
                    #         # print("letter :A",end='\r',flush=True)
                    #         my_lable.config(text="letter: A")
                    #     if self.classes[int(results2[0][0])] == 'correct':
                    #         # print("letter :B",end='\r',flush=True)
                    #         my_lable.config(text="letter: B")
                            
                    #     if self.classes[int(results3[0][0])] == 'correct':
                    #         # print("letter :C",end='\r',flush=True)
                    #         my_lable.config(text="letter: C")
                            
                    #     if self.classes[int(results4[0][0])] == 'correct':
                    #         # print("letter :D",end='\r',flush=True)
                    #         my_lable.config(text="letter: D")
                            
                    # except Exception:
                    #     # print("Can't detect", end='\r', flush=True)
                    #     my_lable.config(text="Can't detect")
                    
                    
                    
                    
                    # img = self.plot_boxes(results1, img)
                    img1 = self.plot_boxes(results1, img1)
                    img2 = self.plot_boxes(results2, img2)
                    img3 = self.plot_boxes(results3, img3)
                    img4 = self.plot_boxes(results4, img4)
                    img_whole = self.plot_boxes(results_whole, img_whole)
                    img_whole = cv2.cvtColor(img_whole, cv2.COLOR_BGR2RGB)
                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
                    # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
                    img_update = ImageTk.PhotoImage(Image.fromarray(img_whole))
                    # img_update2 = ImageTk.PhotoImage(Image.fromarray(img2))
                    # img_update3 = ImageTk.PhotoImage(Image.fromarray(img3))
                    # img_update4 = ImageTk.PhotoImage(Image.fromarray(img4))
                    paneeli_image.configure(image=img_update)
                    # paneeli_image2.configure(image=img_update2)
                    # paneeli_image3.configure(image=img_update3)
                    # paneeli_image4.configure(image=img_update4)
                    paneeli_image.image=img_update
                    # paneeli_image2.image=img_update2
                    # paneeli_image3.image=img_update3
                    # paneeli_image4.image=img_update4
                    paneeli_image.update()
                    # paneeli_image2.update()
                    # paneeli_image3.update()
                    # paneeli_image4.update()

                
                    def typing_answer():
                        try:
                            locker.acquire()
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results1[0][0])] == 'correct':
                                        my_lable.config(text="letter: A")
                                        keyboard.press_and_release('a')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        self.count += 1
                                        self.valid = True
                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results2[0][0])] == 'correct':
                                        my_lable.config(text="letter: B")
                                        keyboard.press_and_release('b')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        self.count += 1
                                        self.valid = True

                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results3[0][0])] == 'correct':
                                        my_lable.config(text="letter: C")
                                        keyboard.press_and_release('c')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        self.count += 1
                                        self.valid = True
                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results4[0][0])] == 'correct':
                                        my_lable.config(text="letter: D")
                                        keyboard.press_and_release('d')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        self.count += 1
                                        self.valid = True
                                    break
                                
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results4[0][0])] != 'correct' and self.classes[int(results3[0][0])] != 'correct' and self.classes[int(results2[0][0])] != 'correct' and self.classes[int(results1[0][0])] != 'correct':
                                        my_lable.config(text="Can't decide")
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        self.count += 1
                                        self.valid = True
                                    break
                            
                            if self.valid == True:
                                print("count: ",self.count,end='\r',flush=True)
                                self.valid = False
    # A try-except block that is used to catch any exception that might occur in the function.
                        except Exception:
                            print("Can't detect", end='\r', flush=True)
                        finally:
                            locker.release()
                    threading.Thread(target=typing_answer).start()


        painike_1=ttk.Button(ikkuna,text="Start",command=otakuva,width=20)
        painike_1.grid(row=2,column=2,pady=10,padx=10)
        # painike_1.config(height=10,width=10)

        #toggle button
        # toggle = ttk.Checkbutton(ikkuna, text="Start correcting", style="Switch.TCheckbutton")
        # toggle.grid(row=3,column=2,pady=10,padx=10)

        
        sv_ttk.set_theme("dark")

        ikkuna.mainloop()


# Creating a new process for the green_cell function
if __name__ == '__main__':
    
    detection = ObjectDetection()
    detection()
