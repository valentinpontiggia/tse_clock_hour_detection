import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import pygame
import camera



# Affichage de la caméra --> voir le fichier camera.py
def startCamera():
    camWindow = tk.Toplevel(mainwindow)
    camWindow.title("Caméra")
    camWindow.geometry("450x600")
    cam = camera.CameraApp(camWindow,mainwindow)

####################################################################################    
#################### Programme principal : fenêtre d'accueil #######################
mainwindow=tk.Tk()
mainwindow.title("Projet Vision")
mainwindow.geometry("800x600")

# Taille de la fenêtre
start_canvas = tk.Canvas(mainwindow,width=800,height=600)
start_canvas.pack(fill="both", expand=True)

# Image de fond
img = ImageTk.PhotoImage(Image.open("bg.png"))
start_canvas.create_image(10,50,image=img,anchor="nw")

# Titre et définiton du scénario
titre = start_canvas.create_text(400, 50, text="Projet Vision", font=("Verdana",20, "bold"),fill="darkblue")

canvas_rulestext = start_canvas.create_text(640,340, text="Explications projet",font=("Verdana",12),fill="black")


###################### Style des boutons ######################
button_style = {
    "fg": "#902038",     # Couleur du texte
    "font": ("Verdana", 14, "bold"),   # Police en gras, taille 14
    "bd": 3,           # Largeur de la bordure de 3 pixels
    "relief": "ridge", # Type de bordure en relief
    "activebackground": "#2B91FF",    # Couleur de fond lors du survol de la souris
    "activeforeground": "white",      # Couleur du texte lors du survol de la souris
    "highlightcolor": "#F4FA58",      # Couleur de la bordure lors du survol de la souris
    "highlightbackground" : "darkgrey",
    "highlightthickness": 2,          # Epaisseur de la bordure lors du survol de la souris
    "cursor": "hand2"    # Curseur de souris en forme de main pour indiquer l'interactivité
}


###################### Création du bouton Start ######################
start_button = tk.Button(mainwindow, text="START", **button_style, command=startCamera)
start_button_window = start_canvas.create_window(540,140,anchor="nw", window=start_button)

###################### Création du menu ######################
def createMenu():
    menu=tk.Menu(mainwindow)

    options=tk.Menu(menu,tearoff=0)
    options.add_command(label="Option 1",command=None)
    options.add_command(label="option 2",command=None)
    menu.add_cascade(label="Options",menu=options)

    close=tk.Menu(menu,tearoff=0)
    menu.add_cascade(label="Fermer",command=mainwindow.quit)
    mainwindow.config(menu=menu)
createMenu()

mainwindow.mainloop()