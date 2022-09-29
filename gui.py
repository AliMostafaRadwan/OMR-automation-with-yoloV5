import tkinter
from tkinter import *
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import cv2
import mss

def GUI():

    ikkuna=tkinter.Tk()
    ikkuna.title("Example about handy CV2 and tkinter combination...")

    frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    ####
    paneeli_image=tkinter.Label(ikkuna) #,image=img)
    paneeli_image.grid(row=0,column=0,columnspan=1,pady=1,padx=10)
    ####
    paneeli_image2=tkinter.Label(ikkuna) #,image=img)
    paneeli_image2.grid(row=0,column=1,columnspan=1,pady=1,padx=10)
    ####
    paneeli_image3=tkinter.Label(ikkuna) #,image=img)
    paneeli_image3.grid(row=0,column=2,columnspan=1,pady=1,padx=10)
    ####
    paneeli_image4=tkinter.Label(ikkuna) #,image=img)
    paneeli_image4.grid(row=0,column=3,columnspan=1,pady=1,padx=10)
    ####
    message="You can see some \nclassification results \nhere after you add some intelligent  \nadditional code to your combined and handy \n tkinter & CV2 solution!"
    paneeli_text=tkinter.Label(ikkuna,text=message)
    paneeli_text.grid(row=1,column=1,pady=1,padx=10)

    def otakuva():

        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            while "Screen capturing":
                img = np.array(sct.grab(monitor))
                h, w, channels = img.shape
                half2 = h//2
                bottom = img[half2:, :]
                #Update the image to tkinter...
                bottom1 = img[half2:, :]
                bottom2 = img[half2:, :]
                bottom3= img[half2:, :]
                bottom4 = img[half2:, :]
                img4 = bottom4[230:390, 740:900]
                img3 = bottom3[230:390, 900:1050]
                img2 = bottom2[230:390, 1050:1200]
                img1 = bottom1[230:390, 1190:1380]
                
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                
                img_update = ImageTk.PhotoImage(Image.fromarray(img1))
                img_update2 = ImageTk.PhotoImage(Image.fromarray(img2))
                img_update3 = ImageTk.PhotoImage(Image.fromarray(img3))
                img_update4 = ImageTk.PhotoImage(Image.fromarray(img4))
                
                
                paneeli_image.configure(image=img_update)
                paneeli_image2.configure(image=img_update2)
                paneeli_image3.configure(image=img_update3)
                paneeli_image4.configure(image=img_update4)
                
                paneeli_image.image=img_update
                paneeli_image2.image=img_update2
                paneeli_image3.image=img_update3
                paneeli_image4.image=img_update4
                
                paneeli_image.update()
                paneeli_image2.update()
                paneeli_image3.update()
                paneeli_image4.update()
                

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break

    def lopeta():
        cv2.destroyAllWindows()
        print("Stopped!")

    painike_korkeus=10
    painike_1=tkinter.Button(ikkuna,text="Start",command=otakuva,height=5,width=20)
    painike_1.grid(row=1,column=0,pady=10,padx=10)
    painike_1.config(height=1*painike_korkeus,width=20)

    painike_korkeus=10
    painike_1=tkinter.Button(ikkuna,text="Stop",command=lopeta,height=5,width=20)
    painike_1.grid(row=1,column=2,pady=10,padx=10)
    painike_1.config(height=1*painike_korkeus,width=20)

    ikkuna.mainloop()
GUI()