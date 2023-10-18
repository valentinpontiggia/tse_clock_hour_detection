import cv2
import tkinter as tk
from PIL import Image, ImageTk
class CameraApp:
    def __init__(self, master,master2):
        self.master = master
        self.master2 = master2
        self.camera = cv2.VideoCapture(0)

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
        self.btn_classify = tk.Button(self.master, text="Analyser", command=None)
        self.btn_classify.pack()    
        _, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.save("reponse.jpg")
        