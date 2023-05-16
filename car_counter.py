from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
import math
import time
from sort import *

cap = cv2.VideoCapture(r'data\traffic.mp4')#for videos
model =YOLO(r'weights\yolov8n.pt')
mask = cv2.imread(r'data\mask.png')

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [190,160,460,160]
totalcount = []

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
             "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", 
             "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", 
             "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

def count_car():
    while cap.isOpened():
        success,img = cap.read()
        imgregion = cv2.bitwise_and(img,mask)  # for adding the mask in video
        if not success:
            break
        results = model(imgregion,stream=True)
        detections = np.empty((0, 5))
        for r in results:
            boxes=r.boxes
            for box in boxes:
                #snippet for bounding box
                x1,y1,x2,y2 = box.xyxy[0]     
                x1,y1,x2,y2 = int(x1) , int(y1) , int(x2) , int(y2)   
                w,h = x2-x1 , y2-y1
                #confidence interval
                conf = math.ceil((box.conf[0]*100))/100
                #class names
                cls = int(box.cls[0])
                currentclass = classNames[cls]

                if currentclass == 'car' or currentclass=='bus' or currentclass=='truck'\
                    or currentclass =='motorbike' and conf>0.3:

                    #cvzone.putTextRect(img , f'{conf} {currentclass}' ,(max(0,x1),max(20,y1)),scale=0.6,thickness=1,
                                #  offset=3)
                    #cvzone.cornerRect(img,(x1,y1,w,h),l=6,rt=5)
                    currentArray = np.array([x1,y1,x2,y2 , conf]) ##getting the points of rectangle 
                    detections = np.vstack((detections , currentArray)) #stacking the two array

        resulttracker = tracker.update(detections)
        cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),5)   #drawing a line at (180,160,460,160)
        for result in resulttracker:
            x1,y1,x2,y2,id = result   #geting points for rectangle and car
            x1,y1,x2,y2 = int(x1) , int(y1) , int(x2) , int(y2) 
            #print(result)
            w,h = x2-x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=6,rt=2,colorR=(255,0,255))
            cvzone.putTextRect(img , f'{int(id)}' ,(max(0,x1),max(25,y1)),scale=2,thickness=2,offset=4)
            
            cx,cy =  x1+w//2 , y1+h//2  ##getting the center of oject detected
            cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)

            if limits[0]<cx<limits[2] and limits[1]-10 <cy< limits[1]+10:
                if totalcount.count(id) == 0:
                    totalcount.append(id)
                    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)   #drawing a line at (180,160,460,160)

        cvzone.putTextRect(img , f'Total count:{len(totalcount)}' ,(50,50))
        cv2.imshow(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows() 