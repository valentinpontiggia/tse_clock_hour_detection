import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
class CameraApp:
    def __init__(self, master,master2):
        self.master = master
        self.master2 = master2
        self.camera = cv2.VideoCapture(0)
        self.isButtonVisible = False

        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

        self.btn_capture = tk.Button(self.master, text="Capture", command=self.capture)
        self.btn_capture.pack()
        
        self.update_stream()

# Cette méthode lit une image du flux vidéo de la caméra, la convertit de BGR à RGB 
# (puisque OpenCV lit les images en BGR tandis que tkinter les attend en RGB), 
# puis l'affiche sur le Canvas
    def update_stream(self):
        _, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.imgtk = imgtk
        self.master.after(10, self.update_stream)

# Cette méthode est appelée lorsqu'on clique sur le bouton Capture. Elle capture une 
# image à partir du flux de la caméra, la convertit de BGR à RGB, puis sauvegarde cette 
# image dans un fichier nommé "reponse.jpg".
    def capture(self):
        if self.isButtonVisible == False :
            self.btn_classify = tk.Button(self.master, text="Analyser", command=self.analyse)
            self.btn_classify.pack()
            self.isButtonVisible = True    
        _, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.save("reponse.jpg")
        
    def analyse(self):
        img = cv2.imread('reponse.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img_blurred = cv2.GaussianBlur(img_gray,(5,5),0)
        
        edges = cv2.Canny(img_blurred, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        dilatation = cv2.dilate(edges,kernel,iterations = 1)
        erosion = cv2.erode(dilatation,kernel,iterations = 1)
        
        mask = img_gray>200
        erosion[mask]=0
        cv2.imwrite("analyse.png",erosion)