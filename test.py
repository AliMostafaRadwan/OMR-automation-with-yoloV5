import time
import cv2
from matplotlib import pyplot as plt
import tkinter as Tkinter
from PIL import Image, ImageTk
import sys

img = cv2.imread("imgTrain\0_img1_n.png")

#- display on OpenCV window -
def displayAtOpenCV():
    cv2.namedWindow('imageWindow', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('imageWindow',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#- display with matplotlib -
def displayAtPyplot():
    plt.figure().canvas.set_window_title("Hello Raspberry Pi")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
#- display on Tkinter -
def displayAtThinter():
    root = Tkinter.Tk() 
    b,g,r = cv2.split(img) 
    img2 = cv2.merge((r,g,b))
    img2FromArray = Image.fromarray(img2)
    imgtk = ImageTk.PhotoImage(image=img2FromArray) 
    Tkinter.Label(root, image=imgtk).pack() 
    root.mainloop()

def displayUsage():
    print("usage: ")
    print("python pyCV_picam.py 1 - display wiyh OpenCV window")
    print("python pyCV_picam.py 2 - display with matplotlib")
    print("python pyCV_picam.py 3 - display with Tkinter")

displayAtThinter()


# if len(sys.argv) != 2:
#     displayUsage()
#     sys.exit()
    
# opt = sys.argv[1]

# if opt=="1":
#     print("display wiyh OpenCV window")
#     capturePiCam()
#     displayAtOpenCV()
# elif opt=="2":
#     print("display with matplotlib")
#     capturePiCam()
#     displayAtPyplot()
# elif opt=="3":
#     print("display with Tkinter")
#     capturePiCam()
#     displayAtThinter()
# else:
#     displayUsage()