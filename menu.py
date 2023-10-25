import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
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
mainwindow.geometry("600x600")

# Taille de la fenêtre
start_canvas = tk.Canvas(mainwindow,width=600,height=600)
start_canvas.pack(fill="both", expand=True)

# Image de fond
img = ImageTk.PhotoImage(Image.open("bg.jpg"))
start_canvas.create_image(0,0,image=img,anchor="nw")

# Titre et textes
titre = start_canvas.create_text(170, 250, text="Projet Vision", font=("Verdana",20, "bold"),fill="white")

canvas_rulestext = start_canvas.create_text(160,90, text="Plus de problèmes pour lire l'heure ! \nGrâce à notre incroyable Clock \nAnalyser, vous serez toujours à \nl'heure pour votre rendez-vous !",font=("Verdana",11),fill="lightgray")

canvas_text = start_canvas.create_text(100,400, text="Projet Vision FISE3 2023\n\nAlban Lemiere\nEnzo Lievoux\nJulien Henry\nValentin Pontiggia\n\nTélécom Saint-Etienne",font=("Verdana",10),fill="lightgray")

###################### Style des boutons ######################
button_style = {
    "fg": "white",     # Couleur du texte
    "font": ("Verdana", 14, "bold"),   # Police en gras, taille 14
    "bg": "#A83219",
    "bd": 0,
    "cursor" : "hand2",
}


###################### Création du bouton Start ######################
start_button = tk.Button(mainwindow, text="START", **button_style, command=startCamera)
start_button_window = start_canvas.create_window(327,250,anchor="nw", window=start_button)

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