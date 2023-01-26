import torch
import numpy as np
import cv2
import mss
import threading
import keyboard
import time
from tkinter import *
from PIL import Image, ImageTk
import customtkinter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.A_count = 0
        self.B_count = 0
        self.C_count = 0
        self.D_count = 0
        print("\n\nDevice Used:",self.device)


    """
    > Loads the model from the path specified in the function
    :return: The model is being returned.
    """

    def load_model(self):
        return torch.hub.load('model_repo\yolov5', model='custom', path='src/best.pt', source='local', force_reload=True)

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
                # cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, bgr, 1)
                # cv2.putText(frame, f'{row[4]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr2, 1)
        return frame
# A global variable that is used to lock the thread.
    global locker
    locker = threading.Lock()
    def __call__(self):
        threading.Thread(target=self.main).start()


    def main(self):  # sourcery skip: merge-nested-ifs
        
        def animate(i):
            x_vals = ['A', 'B', 'C', 'D']
            y_vals = [self.A_count, self.B_count, self.C_count, self.D_count]
            plt.cla()  # clear the current axes
            plt.bar(x_vals, y_vals)
        
        
        
        root=customtkinter.CTk()
        root.title("automation for RemarkOffice software with ML made by ali mostafa")
        root.geometry("700x700")
        root.resizable(0,0)
        root.grid_rowconfigure(1, weight=1, minsize=200)
        root.grid_columnconfigure(0, weight=1, minsize=200)
        
        
        frame_1 = customtkinter.CTkFrame(master=root, width=250, height=240, corner_radius=15)#buttons
        frame_1.grid(row=4, column=0, padx=20, pady=20, sticky="nsew")#buttons
        frame_1.grid_columnconfigure(0, weight=1)#buttons
        frame_1.grid_columnconfigure(1, weight=1)#buttons
        
        
        frame_2 = customtkinter.CTkFrame(master=root, width=200, height=190, corner_radius=1)#video feed
        frame_2.grid(row=1, column=0, padx=20, pady=40, sticky="ew")#video feed
        frame_2.grid_columnconfigure(0, weight=1)#video feed
        frame_2.grid_columnconfigure(1, weight=1)#video feed
        
        
        frame_3 = customtkinter.CTkFrame(master=root, width=200, height=40, corner_radius=15)#letters,counter
        frame_3.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")#letters,counter
        frame_3.grid_columnconfigure(0, weight=1)#letters,counter
        frame_3.grid_columnconfigure(1, weight=1)#letters,counter
        
        
        frame_4 = customtkinter.CTkFrame(master=root, width=200, height=40, corner_radius=15)#plot
        frame_4.grid(row=3, column=0, padx=25, pady=20, sticky="nsew")#plot
        frame_4.grid_columnconfigure(0, weight=1)#plot
        frame_4.grid_columnconfigure(1, weight=1)#plot
        

        plt.gcf().set_size_inches(8, 2.3)
        canvas = FigureCanvasTkAgg(plt.gcf(), master=frame_4)
        canvas.get_tk_widget().grid(column=0, row=1)
        ani = FuncAnimation(plt.gcf(), animate, interval=100)



        image_panel=customtkinter.CTkLabel(frame_2,text=' ') #,image=img)
        image_panel.grid(row=0,column=2,sticky="nsew",padx=10)

        my_lable = customtkinter.CTkLabel(frame_3, text=" ",text_font=("Arial", 25),text_color="red")
        my_lable.grid(row=1, column=0)
        
        my_count = customtkinter.CTkLabel(frame_3, text=" ",text_font=("Arial", 25),text_color="white")
        my_count.grid(row=1, column=1)

        def maingui():
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
                    
                    
                    if lang_check_button.get():# language checkbox (English)
                        img1 = bottom4[230:390, 740:900]
                        img2 = bottom3[230:390, 900:1050]
                        img3 = bottom2[230:390, 1050:1200]
                        img4 = bottom1[230:390, 1190:1380]

                    results1 = self.score_frame(img1)
                    results2 = self.score_frame(img2)
                    results3 = self.score_frame(img3)
                    results4 = self.score_frame(img4)
                    results_whole = self.score_frame(img_whole)

                    def typing_answer():
                        try:
                            locker.acquire()
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results1[0][0])] == 'correct':
                                        keyboard.press_and_release('a')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        my_lable.configure(text="letter: A")
                                        self.count += 1
                                        self.A_count += 1
                                        my_count.configure(text=self.count)
                                        self.valid = True
                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results2[0][0])] == 'correct':
                                        keyboard.press_and_release('b')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        my_lable.configure(text="letter: B")
                                        self.count += 1
                                        self.B_count += 1
                                        my_count.configure(text=self.count)
                                        self.valid = True

                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results3[0][0])] == 'correct':
                                        keyboard.press_and_release('c')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        my_lable.configure(text="letter: C")
                                        self.count += 1
                                        self.C_count += 1
                                        my_count.configure(text=self.count)
                                        self.valid = True
                                    break
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results4[0][0])] == 'correct':
                                        keyboard.press_and_release('d')
                                        time.sleep(0.1)
                                        keyboard.press_and_release('enter')
                                        time.sleep(1)
                                        my_lable.configure(text="letter: D")
                                        self.count += 1
                                        self.D_count += 1
                                        my_count.configure(text=self.count)
                                        self.valid = True
                                    break
                                
                            for _ in range(1):
                                if self.valid == False:
                                    if self.classes[int(results4[0][0])] != 'correct' and self.classes[int(results3[0][0])] != 'correct' and self.classes[int(results2[0][0])] != 'correct' and self.classes[int(results1[0][0])] != 'correct':
                                        my_lable.configure(text="Can't detect")
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
                    
                    
                    # img = self.plot_boxes(results1, img)
                    img1 = self.plot_boxes(results1, img1)
                    img2 = self.plot_boxes(results2, img2)
                    img3 = self.plot_boxes(results3, img3)
                    img4 = self.plot_boxes(results4, img4)
                    img_whole = self.plot_boxes(results_whole, img_whole)
                    img_whole = cv2.cvtColor(img_whole, cv2.COLOR_BGR2RGB)
                    img_update = ImageTk.PhotoImage(Image.fromarray(img_whole))
                    image_panel.configure(image=img_update)
                    image_panel.image=img_update
                    image_panel.pack()
                    image_panel.update()

        button_1 = customtkinter.CTkButton(master=frame_1, text="Start", height=32,
                                                compound="right", command=maingui)
        button_1.grid(row=2, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        
        lang_check_button = customtkinter.CTkCheckBox(master=frame_1,
                                                        text="English")
        lang_check_button.grid(row=1, column=1, pady=10, padx=20,sticky="nsew")



        root.mainloop()


# Creating a new process for the green_cell function
if __name__ == '__main__':
    
    detection = ObjectDetection()
    detection()
