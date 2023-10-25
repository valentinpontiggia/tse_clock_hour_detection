import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.morphology import skeletonize
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
    
        
    def analyse(self):
        # Read image
        img = cv2.imread('apprentissage/horloges/clock12.png')
        hh, ww = img.shape[:2]

        # convert to gray
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

        # invert so shapes are white on black background
        thresh = 255 - thresh

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
        
        methodKey = 2
        
         # Prepare some lists to store every coordinate of the detected lines:
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        
        if methodKey ==1:
            # Store and draw the lines:
            for [currentLine] in lines:

                # First point:
                x1 = currentLine[0]
                y1 = currentLine[1]
                X1.append(x1)
                Y1.append(y1)

                # Second point:
                x2 = currentLine[2]
                y2 = currentLine[3]
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
        cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)


        # save results
        cv2.imwrite('apprentissage/clock_thresh.jpg', thresh)
        cv2.imwrite('apprentissage/clock_arms.jpg', (255*arms).clip(0,255).astype(np.uint8))
        cv2.imwrite('apprentissage/clock_arms_thin.jpg', arms_thin)
        cv2.imwrite('apprentissage/clock_lines.jpg', result)

        cv2.imshow('thresh', thresh)
        cv2.imshow('arms', (255*arms).clip(0,255).astype(np.uint8))
        cv2.imshow('arms_thin', arms_thin)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()