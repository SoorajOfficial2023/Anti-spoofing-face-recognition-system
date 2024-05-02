from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

##########################
classID = 0 #0 is fake and 1 is real
confidence = 0.8      #to make more accurate detection
floatingPoint = 6 #to round the decimel figure
blurThreshold = 35 #larger value have more focused

debug = False

offsetPercentageW = 10
offsetPercentageH = 20
camWidth,camHight = 640,480

save = True   # is a parameter, sometimes we want to debug not save again and again
outputFolderPath = 'Dataset/dataCollect'
#########################

cap = cv2.VideoCapture(2)
cap.set(3,camWidth)
cap.set(4,camHight)
detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()  #The saved image don't need bbox & label on it. so that the imgOut will display and a copy of it with bbox will display as img
    img, bboxs = detector.findFaces(img,draw=False)
    listBlur = [] #True False values indicating if the faces are blur or not
    listInfo = [] #The normalized values and class name for the labelled txt file
    if bboxs:
        for bbox in bboxs:
            x,y,w,h = bbox['bbox']
            score = bbox['score'][0]
            # print(x,y,w,h)
            
            #------------ check the score --------
            if score > confidence:
                
                
                #------- Adding an offset to the face detected -------enlarge the bbox to completely track the face
                offsetW = (offsetPercentageW/100)*w
                x = int(x - offsetW)
                w = int(w+offsetW*2)
                
                offsetH = (offsetPercentageH/100)*h
                y = int(y - offsetH*3)
                h = int(h+offsetH*3.5)
                
                #---------To avoid values below 0--------- #do not x,y,w,h to have value 0,this is to avoid corrept the image
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0
                  
                #------- Find bluriness-------check the blurriness persentage
                imgFace = img[y:y+h,x:x+w]
                # cv2.imshow('face',imgFace)
                blurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True) #it means it's not blurry
                else:
                    listBlur.append(False)
                    
                #----------Normalize values-------
                ih,iw,_ = img.shape
                xc,yc = x+w/2,y+h/2   #x,y axis in center
                # print(xc,yc)
                xcn,ycn  = round(xc/iw,floatingPoint), round(yc/ih,floatingPoint)  #center normalize value
                wn,hn = round(w/iw,floatingPoint), round(h/ih,floatingPoint) 
                # print(xcn,ycn,wn,hn)
                
                #---------To avoid values above 1--------- #to avoid corrept data
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1
                
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                  
                
                #----------Drawing --------
                cv2.rectangle(imgOut,(x,y,w,h),(255,0,0),3)
                cvzone.putTextRect(imgOut,f"Score: {int(score*100)}% Blur Value: {blurValue}",(x,y-20),
                                   scale=1,thickness=2)
                
                if debug:
                    cv2.rectangle(img,(x,y,w,h),(255,0,0),3)
                    cvzone.putTextRect(img,f"Score: {int(score*100)}% Blur Value: {blurValue}",(x,y-20),
                                   scale=1,thickness=2)
        
        
        #------to save--------
        if save:
            if all(listBlur) and listBlur != []:
                #------to save image----
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0]+timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)
                #-------------save labeled text file-------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt",'a')
                    f.write(info)
                    f.close()
                
                
    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
