import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import math
from skimage.morphology import skeletonize
class CameraApp:
    def __init__(self, master,master2):
        self.master = master
        self.master2 = master2
        self.camera = cv2.VideoCapture(0)
        self.interval = 10
        self.isButtonVisible = False
        self.analyse_continuously = False
        self.analyse_interval = 500 # Interval de 1 sec
    
        self.canvas = tk.Canvas(self.master, width=500, height=480)
        self.canvas.grid(row=0, column=0, sticky='nw')

        self.text_canva = tk.Canvas(self.master, width=280, height=480)
        self.text_canva.grid(row=0, column=1, sticky='ne')
        self.heureText = self.text_canva.create_text(150,100,text="Heure détectée :", font=("Imprint MT Shadow",14, "bold"))
        self.heureDetect = self.text_canva.create_text(150,150,text="12 : 00", font=("Imprint MT Shadow",18, "bold"))

        self.btn_capture = tk.Button(self.master, text="Capture", command=self.capture)
        self.btn_capture.grid(row=1, column=0, columnspan=1, pady=5, sticky='n')
        self.btn_continuous_analysis = tk.Button(self.master, text="Analyse Continue", command=self.toggle_continuous_analysis)
        self.btn_continuous_analysis.grid(row=1, column=0, columnspan=1, pady=45, sticky='n')
        
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

        if self.analyse_continuously:
            self.analyse()
        
        self.master.after(self.interval,self.update_stream)

    def toggle_continuous_analysis(self):
        # Méthode pour activer/désactiver l'analyse continue
        if self.analyse_continuously:
            self.interval = 10
        else :
            self.interval = self.analyse_interval
        self.analyse_continuously = not self.analyse_continuously

    def start_continuous_analysis(self):
        self.analyse_continuously = True

    def stop_continuous_analysis(self):
        self.analyse_continuously = False
    
# Cette méthode est appelée lorsqu'on clique sur le bouton Capture. Elle capture une 
# image à partir du flux de la caméra, la convertit de BGR à RGB, puis sauvegarde cette 
# image dans un fichier nommé "reponse.jpg".
    def capture(self):
        if self.isButtonVisible == False :
            self.btn_classify = tk.Button(self.master, text="Analyser", command=self.analyse)
            self.btn_classify.grid(row=1, column=0, columnspan=1, pady=80, sticky='n')
            self.isButtonVisible = True    
        _, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.save("reponse.jpg")
    
       
    #def analyse(self):
    #    img = cv2.imread('apprentissage/horloges/clock44.png')
    #   img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #  
    #    img_blurred = cv2.GaussianBlur(img_gray,(5,5),0)
    #    
    #    edges = cv2.Canny(img_blurred, 50, 150)
    #    kernel = np.ones((5,5), np.uint8)
    #    dilatation = cv2.dilate(edges,kernel,iterations = 1)
    #    erosion = cv2.erode(dilatation,kernel,iterations = 1)
    #    
    #    #mask = img_gray>200
    #    #erosion[mask]=0
    #    cv2.imwrite("apprentissage/analyses/clock44.png",erosion)
    
    def findCenter(self, img_thresh):
        edges = cv2.Canny(img_thresh, 30, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtrage des contours basé sur la superficie
        seuil = 10000
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > seuil]
        
        img_with_contours = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 255, 0), 2)

        # Affichage de l'image avec les contours
        cv2.imshow('Contours', img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        M = cv2.moments(filtered_contours[0])
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        center = []
        center.append(center_x)
        center.append(center_y)
        return center
    
    def is_point_near_center(self, point, center, distance_threshold):
        # Fonction pour vérifier si un point est suffisamment proche du centre
        distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return distance < distance_threshold
    
    def calculate_angle(self, center, point):
        x, y = point[0] - center[0], point[1] - center[1]
        angle_rad = math.atan2(x, -y)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    def angle_to_hour(self, angle):
    # Convertir l'angle en heures (360 degrés correspondent à 12 heures)
        hour = angle * 12 / 360

        # Gérer les cas spéciaux
        if hour < 0:
            hour += 12
        elif hour > 12:
            hour -= 12

        return int(hour)
    
    def angle_to_minute(self, angle):
    # Convertir l'angle en heures (360 degrés correspondent à 60 minutes)
        minute = round(angle * 60 / 360)
        if minute < 0:
            minute += 60
        elif minute > 60:
            minute -= 60
        return minute

        
    def analyse(self):
        # Read image
        img = cv2.imread('apprentissage/horloges/clock44.png')
        hh, ww = img.shape[:2]

        # convert to gray
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

        # invert so shapes are white on black background
        thresh = 255 - thresh
        center = self.findCenter(thresh)
        print("Centre : ", center)
        center_img = img.copy()
        cv2.circle(center_img, (center[0], center[1]), 5, (0, 255, 0), -1)
        cv2.imwrite('apprentissage/clock_center.jpg', center_img)

        cv2.imshow('Center', center_img)

        # get contours and save area
        cntrs_info = []
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        index=0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            cntrs_info.append((index,area))
            index = index + 1

        # sort contours by area
        def takeSecond(elem):
            return elem[1]
        cntrs_info.sort(key=takeSecond, reverse=True)

        # get third largest contour
        arms = np.zeros_like(thresh)
        index_third = cntrs_info[2][0]
        cv2.drawContours(arms,[contours[index_third]],0,(1),-1)

        #arms=cv2.ximgproc.thinning(arms)
        arms_thin = skeletonize(arms)
        arms_thin = (255*arms_thin).clip(0,255).astype(np.uint8)

        # Création image noire pour y poser les lignes dessus
        blank = np.zeros([hh,ww])

        # get hough lines and draw on copy of input
        result = img.copy()
        lineThresh = 15
        minLineLength = 20
        maxLineGap = 100
        max
        lines = cv2.HoughLinesP(arms_thin, 1, np.pi/180, lineThresh, None, minLineLength, maxLineGap)


        #for [line] in lines:
        #    x1 = line[0]
        #    y1 = line[1]
        #    x2 = line[2]
        #    y2 = line[3]
        #    cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)
        
        methodKey = 1
        
         # Prepare some lists to store every coordinate of the detected lines:
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        distance_threshold = 40
        if methodKey ==1:
            # Store and draw the lines:
            for [currentLine] in lines:
                print("test")
                x1 = currentLine[0]
                y1 = currentLine[1]
                x2 = currentLine[2]
                y2 = currentLine[3]
                
                
                if (self.is_point_near_center((x1, y1), center, distance_threshold) or
                    self.is_point_near_center((x2, y2), center, distance_threshold)):
                    # First point:
                    
                    X1.append(x1)
                    Y1.append(y1)

                    # Second point:
                    X2.append(x2)
                    Y2.append(y2)
        
        elif methodKey == 2:
            # Utilisation des Kmeans pour épurer les lignes affichées
            X1 = np.array(X1)
            Y1 = np.array(Y1)
            X2 = np.array(X2)
            Y2 = np.array(Y2)

            X1dash = X1.reshape(-1,1)
            Y1dash = Y1.reshape(-1,1)
            X2dash = X2.reshape(-1,1)
            Y2dash = Y2.reshape(-1,1)

            # Stack the data
            Z = np.hstack((X1dash, Y1dash, X2dash, Y2dash))
            print(Z)

            # K-means operates on 32-bit float data:
            floatPoints = np.float32(Z)

            # Set the convergence criteria and call K-means:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # Set the desired number of clusters
            K = 2
            ret, label, center = cv2.kmeans(floatPoints, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Draw the lines:
        for i in range(len(X1)):
            print(f"Len x1 {len(X1)}")
            cv2.line(blank, (X1[i],Y1[i]), (X2[i],Y2[i]), (255,255,255), 2)
            if not (self.is_point_near_center((X1[i], Y1[i]),center,distance_threshold)):
                point_on_line = (X1[i], Y1[i])
                cv2.circle(blank,(point_on_line),8,(255, 0, 0))
                angle_hours = self.calculate_angle(center, point_on_line)
                length_1 = np.sqrt((point_on_line[0]-center[0])**2 + (point_on_line[1]-center[0])**2)
            if not (self.is_point_near_center((X2[i], Y2[i]),center,distance_threshold)):
                point_on_line_2 = (X2[i], Y2[i])
                cv2.circle(blank,(point_on_line_2),8,(255, 0, 0))
                angle_minutes = self.calculate_angle(center, point_on_line_2)
                length_2 = np.sqrt((point_on_line_2[0]-center[0])**2 + (point_on_line_2[1]-center[0])**2)
        if length_1 > length_2:
            angle_minutes = self.calculate_angle(center, point_on_line)
            angle_hours = self.calculate_angle(center, point_on_line_2)

        heure = self.angle_to_hour(angle_hours)
        minutes = self.angle_to_minute(angle_minutes)
        if heure<10 :
            heure = "0"+str(heure)
        if minutes<10 :
            minutes = "0"+str(minutes)
        hour_detected_text = str(heure)+" : "+ str(minutes)
        self.text_canva.itemconfig(self.heureDetect, text=hour_detected_text)

        # save results
        cv2.imwrite('apprentissage/clock_thresh.jpg', thresh)
        cv2.imwrite('apprentissage/clock_arms.jpg', (255*arms).clip(0,255).astype(np.uint8))
        cv2.imwrite('apprentissage/clock_arms_thin.jpg', arms_thin)
        cv2.imwrite('apprentissage/clock_lines.jpg', blank)

        #cv2.imshow('thresh', thresh)
        #cv2.imshow('arms', (255*arms).clip(0,255).astype(np.uint8))
        #cv2.imshow('arms_thin', arms_thin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()