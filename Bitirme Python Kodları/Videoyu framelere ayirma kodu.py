from tkinter import *
from tkinter import filedialog
import cv2
import os

root = Tk()
root.geometry('300x300')    

l = Label(root, text = "Firstly choose a video then")
l1 = Label(root, text = "click button to convert video to frames")
l.pack()
l1.pack()

folder_path = StringVar()
folder_path2 = StringVar()

menubar = Menu(root)
root.config(menu=menubar)

subMenu = Menu(menubar, tearoff=0)


def browse_file():
    global filename_path
    filename_path = filedialog.askopenfilename()
    folder_path.set(filename_path)
    print(filename_path)

def browse_file2():
    global filename_path2
    filename_path2 = filedialog.askdirectory()
    folder_path2.set(filename_path2)
    print(filename_path2)

menubar.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Open file to choose video", command=browse_file)
subMenu.add_command(label="Select folder to save frames", command=browse_file2)


def buttonfunction():
    count = 0
    videoFile = filename_path 
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = 1.5 #frame rate
    x=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret != True):
            break
        filename ="frame%d.jpg" % count;count+=1
        dirname = filename_path2
        os.chdir(dirname)
        cv2.imwrite(filename, frame)
    cap.release()
    print ("Done!")
    

b = Button(root, text = "Click me", command=buttonfunction)
b.pack()

root.mainloop()



