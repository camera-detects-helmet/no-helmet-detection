import torch
import cv2 as cv
import numpy as np
from tracker import *
from LoadSourceFile import *
import argparse

def POINTS(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        colorBGR = [x,y]
        print(colorBGR)
    cv.namedWindow('ROI')
    cv.setMouseCallback('ROI',POINTS)
        

def ObjectDetectorTrack(source=0,location='test_cameras'):

    model_rider = torch.hub.load('ultralytics/yolov5', 'custom', path='models/new_models/RIDER_best.pt')  # local model
    model_helmet = torch.hub.load('ultralytics/yolov5', 'custom', path='models/helmet_9_2_5m6_64_01.pt')  # local model
    model_helmet.conf  = 0.4
    model_rider.conf = 0.6
    cap = LoadSourceFile(source=0)
    cap.config(640,480)
    
    count = 0
    num_of_no_helmet = 0
    num_of_rider = 0
    
    


    while True:
        frame = cap.read()
        

        result = []
        count += 1
        
        if count % 3 != 0:
            continue
    
        if frame is not None:
            ##put text into buttom middle frame
            results = model_rider(frame)
            
            rider_crop = []
            
            result_rider = results.pandas().xyxy[0].iterrows()
            num_of_rider = len(results.pandas().xyxy[0])
            cap_width , cap_height = cap.cap.get(3),cap.cap.get(4)
            for index,row in result_rider:
                
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                
                d =(row['name'])
                confidence = row['confidence']
                result.append([d,confidence])
                # print(f'[INFO] [RIDER] name: {d} , confidence : {round(confidence,3)} , x1: {x1} , y1: {y1} , x2: {x2} , y2: {y2}')
                if 'rider' in d:                
                    ##crop top rider and show
                    top_xmin , top_ymin , top_xmax , top_ymax = x1,y1,x2,int(y1+(y2-y1)/2)
                    crop_rider_top = frame[top_ymin:top_ymax,top_xmin:top_xmax]
                    cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    cv.putText(frame, f'{d} {round(confidence,2)}', (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
                    
                    rider_crop.append([x1, x2, y1, y2])
                    # cv.rectangle(frame,(top_xmin,top_ymin),(top_xmax,top_ymax),(0,255,0),3)
            
            for [left, right, top, bottom] in rider_crop:
                
                
                cropImage = frame[top:bottom,left:right]
                top_xmin , top_ymin , top_xmax , top_ymax = 0,0,cropImage.shape[1],int(cropImage.shape[0]/2)
                cropTopImage = cropImage[top_ymin:top_ymax,top_xmin:top_xmax]
                
                results_helemt = model_helmet(cropTopImage)
                num_of_no_helmet = len(list(filter(lambda x: 'no_helmet' in x, results_helemt.pandas().xyxy[0]['name'])))
                for index,row in results_helemt.pandas().xyxy[0].iterrows():
                    
                    x1 = int(row['xmin']) + left
                    y1 = int(row['ymin']) + top
                    x2 = int(row['xmax']) + left
                    y2 = int(row['ymax']) + top
                    
                    d =(row['name'])
                    confidence = row['confidence']
                    
                    result.append([d,confidence])
                    if 'no_helmet' in d:
                        # cv.rectangle(cropTopImage,(x1,y1),(x2,y2),(0,0,255),3)
                        cv.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                        cv.putText(frame,'No Helmet',(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                        
                    else:
                        # cv.rectangle(cropTopImage,(x1,y1),(x2,y2),(0,255,0),3)
                        cv.rectangle(frame,(x1,y1),(x2,y2),(204,102,0),3)
                        cv.putText(frame,'Helmet',(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(204,102,0),2)
                
                # cv.imshow(d,cropTopImage)
                # cv.putText(frame,f'Rider',(left,top-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                
            # cv.rectangle(frame,(int(cap.cap.get(3)/3)-10,frame.shape[0]-100),(int(cap.cap.get(3)/3)+750,frame.shape[0]),(0,0,0),-1)        
            # cv.putText(frame,f'Rider: {num_of_rider} No-helmet {num_of_no_helmet}',(int(cap.cap.get(3)/3),frame.shape[0]-10),cv.FONT_HERSHEY_SIMPLEX,2,(1,255,0),2)

            for [d,confidence] in result:
                print(f'[INFO] => name: {d} , confidence : {round(confidence,3)} ')
        
            # print(f'💡 > Rider: {num_of_rider} No-helmet {num_of_no_helmet}')
            cv.imshow('ROI',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.stop()
    cv.destroyAllWindows()

def checkDir(self):
    if not os.path.exists('images'):
        print(" > 📁 Create Directory images")
        os.mkdir('images')
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='Location', default='test_cameras')
    parser.add_argument('--source', help='Camera source',default=0 , type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ObjectDetectorTrack()
    
if __name__ == '__main__':
    main()