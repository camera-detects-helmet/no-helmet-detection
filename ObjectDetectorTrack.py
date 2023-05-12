import torch
import cv2 as cv
import numpy as np
from tracker import *
from LoadSourceFile import *
import argparse
import os
import uuid
import base64
import json
from BytesEncoder import BytesEncoder
from dotenv import load_dotenv
import requests
from datetime import datetime

def POINTS(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        colorBGR = [x,y]
        print(colorBGR)
    cv.namedWindow('ROI')
    cv.setMouseCallback('ROI',POINTS)
        


            


def ObjectDetectorTrack(source,location):

    model_rider = torch.hub.load('ultralytics/yolov5', 'custom', path='models/new_models/RIDER_best.pt')  # local model
    model_helmet = torch.hub.load('ultralytics/yolov5', 'custom', path='models/new_models/HELMET_best.pt')  # local model
    model_helmet.conf  = 0.6
    model_rider.conf = 0.8
    print(source)
    cap = LoadSourceFile(source=source)
    
    count = 0
    num_of_no_helmet = 0
    num_of_rider = 0
    
    width = int(cap.cap.get(3))
    height = int(cap.cap.get(4))
    
    if width >= 1280:
        thickness = 2
        fontScale = 0.8
    else:
        thickness = 1
        fontScale = 0.5
    

    result = []
    capture = []
    
    while True:
        frame = cap.read()
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
                    cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),thickness)
                    cv.putText(frame, f'{d} {round(confidence,2)}', (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), thickness, cv.LINE_AA)
                    
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
                        # cv.rectangle(cropTopImage,(x1,y1),(x2,y2),(0,0,255),1)
                        cv.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),thickness)
                        cv.putText(frame,'No Helmet',(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,fontScale,(0,0,255),thickness)
                        
                        # save_frame(frame,cropImage,location)
                        capture.append([frame,cropImage,location])
                        
                        
                        
                    else:
                        # cv.rectangle(cropTopImage,(x1,y1),(x2,y2),(204,102,0),1)
                        cv.rectangle(frame,(x1,y1),(x2,y2),(204,102,0),thickness)
                        cv.putText(frame,'Helmet',(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,fontScale,(204,102,0),thickness)

            
                    
                # cv.imshow('cropImage',cropTopImage)
                # cv.imshow('crop' , cropImage)
            
            if len(rider_crop) > 0:
                rider_crop.pop(0)
            
            if len(capture) > 0:
                for [frame,cropImage,location] in capture:
                    save_frame(frame, cropImage, location)
                capture.pop(0)
                
                    # save_frame(frame,cropImage,location)
                # cv.putText(frame,f'Rider',(left,top-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                
            # cv.rectangle(frame,(int(cap.cap.get(3)/3)-10,frame.shape[0]-100),(int(cap.cap.get(3)/3)+750,frame.shape[0]),(0,0,0),-1)        
            # cv.putText(frame,f'Rider: {num_of_rider} No-helmet {num_of_no_helmet}',(int(cap.cap.get(3)/3),frame.shape[0]-10),cv.FONT_HERSHEY_SIMPLEX,2,(1,255,0),2)

            for [d,confidence] in result:
                print(f'[FPS] {cap.get_fps()} [INFO] => name: {d} , confidence : {round(confidence,3)} ')
            result.clear()

            
            # print(f'💡 > Rider: {num_of_rider} No-helmet {num_of_no_helmet}')
            cv.imshow('ROI',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.stop()
    cv.destroyAllWindows()
    
def save_frame(main_frame,crop_frame,location):
    checkDir()
    id = str(uuid.uuid4())
    filename_mainframe = os.path.join('images',f'{location}_{id}_mainframe.jpg')
    filename_cropframe = os.path.join('images',f'{location}_{id}_cropframe.jpg')
    cv.imwrite(filename_mainframe,main_frame)
    cv.imwrite(filename_cropframe,crop_frame)
    
    print(f'[INFO] [SAVE] {filename_mainframe}')
    print(f'[INFO] [SAVE] {filename_cropframe}')
    now = datetime.now()
    message = f'\nLocation : {location} \nTime : {now.strftime("%d/%m/%Y %H:%M:%S")}'
    send_line(message, filename_mainframe)
    # send_frame(filename_mainframe,filename_cropframe,location)
    

def send_frame(path_mainframe,path_cropframe,location):
    main_frame = cv.imread(path_mainframe)
    crop_frame = cv.imread(path_cropframe)
    
    _, buffer_mainframe = cv.imencode('.jpg', main_frame)
    _, buffer_cropframe = cv.imencode('.jpg', crop_frame)
    
    jpg_as_text_mainframe = base64.b64encode(buffer_mainframe)
    jpg_as_text_cropframe = base64.b64encode(buffer_cropframe)
    
    
    
    url = os.environ.get('API_URL')
    
    if url is None:
        print('[ERROR] [API_URL] Not found')
        return
    
    payload = json.dumps({
        'location': location,
        'base64DefaultImg':jpg_as_text_mainframe,
        'base64CropImg':jpg_as_text_cropframe
    },cls=BytesEncoder)
    
    headers = {
            'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code == 201:
            print(" > ✅ Send Frame to API")
            os.remove(path_mainframe)
            os.remove(path_cropframe)
        else:
            print(" > ❌ Can't send Frame to API")
            
        print(f'[INFO] [API] {response.text}')
    except Exception as e:
        print(f'[ERROR] [API] {e}')
        

def checkDir():
    if not os.path.exists('images'):
        print(" > 📁 Create Directory images")
        os.mkdir('images')

def send_line(message,path_mainframe):
    URL_LINE_NOTIFY = os.environ.get('LINE_NOTIFY_URL')

    TOKEN = os.environ.get('TOKEN_LINE_NOTIFY')
    if TOKEN is None:
        print('[ERROR] [LINE_TOKEN] Not found')
        
    if URL_LINE_NOTIFY is None:
        print('[ERROR] [LINE_URL] Not found')
    
    file = {'imageFile': open(path_mainframe, 'rb')}
    payload = ({'message': message})
    
    LINE_HEADERS = {'Authorization': 'Bearer ' + TOKEN}
    session = requests.Session()
    
   

    try:
        response = session.post(URL_LINE_NOTIFY, headers=LINE_HEADERS, files=file, data=payload)
        print(response.text.encode('utf8'))
        if response.status_code == 200:
            print(" > ✅ Send Message to LINE")
        else:
            print(" > ❌ Can't send Message to LINE")
    except Exception as e:
        print(f'[ERROR] [LINE] {e}')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='Location', default='test_cameras')
    parser.add_argument('--source', help='Camera source',default=0 )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    print(f'[INFO] [SOURCE] {src}')
    
    if checkDir():
        print(f'[INFO] [DIRECTORY] Create Directory images')

    
    ObjectDetectorTrack(source=src,location=args.location)
    
if __name__ == '__main__':
    load_dotenv()
    
    main()
    