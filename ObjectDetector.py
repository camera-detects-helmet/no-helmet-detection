import base64
import json
from threading import Thread
import time
import uuid
import requests
import torch
import cv2 as cv
import os 
from dotenv import load_dotenv
from BytesEncoder import BytesEncoder
from datetime import datetime
import argparse

class ObjectDetector(object):
    def __init__(self,source=0,location='TestLocation'):
        
        self.source = source
        self.rider_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/new_models/RIDER_best.pt')  # local model
        self.rider_model.conf = 0.9
        self.helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/new_models/HELMET_best.pt')  # local model
        self.helmet_model.conf = 0.5
        
        self.save_status = False
        self.location = location
        
        
        self.num_rider = 0
        self.num_no_helmet = 0
        self.stopped = False
        self.capture = cv.VideoCapture(source)
        load_dotenv()
        
        if not self.capture.isOpened():
            print("Cannot open camera")
            exit()
        
        self.FPS = 1/60
        self.FPS_MS = int(self.FPS * 1000)
        
        
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
    
    def detector_rider(self):
        
        result_rider = self.rider_model(self.frame)
        predictions_rider = result_rider.pandas().xyxy[0].values.tolist()
        lamda_rider = list(filter(lambda x: x[6] == 'rider', predictions_rider))
        rider = len(lamda_rider)
        self.num_rider = rider
        
        if rider > 0:
            for i in lamda_rider:                
                crop_rider = self.frame[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                top_xmin , top_ymin , top_xmax , top_ymax = int(i[0]),int(i[1]),int(i[2]),int(i[1]+(i[3]-i[1])/2)
                crop_rider_top = self.frame[top_ymin:top_ymax,top_xmin:top_xmax]
                
                detect_helmet = self.helmet_model(crop_rider_top)
                predictions_no_helmet = detect_helmet.pandas().xyxy[0].values.tolist()
                lamda_no_helmet = list(filter(lambda x: x[6] == 'no_helmet', predictions_no_helmet))
                no_helmet = len(lamda_no_helmet)
                
                self.num_no_helmet = no_helmet
                if no_helmet > 0:
                    self.save_status = True
                    for k in lamda_no_helmet:
                        
                        self.drawRect(self.frame,int(i[0]),int(i[1]),int(i[2]-i[0]),int(i[3]-i[1]))
                        self.drawRect(crop_rider,int(k[0]),int(k[1]),int(k[2]-k[0]),int(k[3]-k[1]),corner_color=(0, 0, 255), rect_color=(200, 200, 200))
                        
                self.show_frame("rider",crop_rider,20,500)
            self.drawRect(self.frame,int(i[0]),int(i[1]),int(i[2]-i[0]),int(i[3]-i[1]))
            

        self.show_frame("RTSP Frame",self.frame,20,10)
        
        
        print(f' {datetime.now().strftime("%H:%M:%S")}> 🏍 Rider :  {self.num_rider} , , ⛑ no helmet : , {self.num_no_helmet}' )
        
        if self.save_status is True:
            self.save_frame(self.frame,crop_rider)  
            time.sleep(2)          
        
        
                
                
    def save_frame(self,main_frame,crop_frame):
        self.checkDir()
        id = str(uuid.uuid4())
        filename_mainframe = f'images/{self.location}_MainFrame_{id}.jpg'
        filename_cropframe = f'images/{self.location}_CropFrame_{id}.jpg'
        
        cv.imwrite(filename_mainframe, main_frame)
        cv.imwrite(filename_cropframe, crop_frame)
        
        print(" > 📸 Save Frame : " , filename_mainframe)
        print(" > 📸 Save Frame : " , filename_cropframe)
        self.send_frame(filename_mainframe,filename_cropframe)
        self.num_no_helmet = 0
        self.save_status = False
        
    
    def send_frame(self,path_mainframe,path_cropframe):
        
        frame = cv.imread(path_mainframe)
        crop_frame = cv.imread(path_cropframe)
        
        _,buffer_mainframe = cv.imencode('.jpg',frame)
        _,buffer_cropframe = cv.imencode('.jpg',crop_frame)
        
        jpg_mainframe = base64.b64encode(buffer_mainframe)
        jpg_cropframe = base64.b64encode(buffer_cropframe)
        
        os.remove(path_cropframe)
        os.remove(path_mainframe)
        
        url = os.environ.get('API_URL')
        
        if url is None:
            print(" > ❌ API_URL is None")
            return
        
        payload = json.dumps({
            "location": self.location,
            "base64DefaultImg": jpg_mainframe,
            "base64RiderImg": jpg_cropframe
        },cls=BytesEncoder)
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, data=payload, headers=headers,timeout=5)
        if response.status_code == 201:
            print(" > ✅ Send Frame to API")
        else:
            print(" > ❌ Can't send Frame to API")
    
    def drawRect(self,frame,x,y,w,h,corner_size=3, rect_size=2, corner_color=(0, 255, 0), rect_color=(200, 200, 200)):
        edge_len = int(min(w, h) /2 * 0.45)
        cv.rectangle(frame, (x, y), (x + w, y + h), rect_color, rect_size)
        cv.line(frame, (x, y), (x, y + edge_len), corner_color, corner_size)
        cv.line(frame, (x, y), (x + edge_len, y), corner_color, corner_size)
        cv.line(frame, (x + w, y), (x + w, y + edge_len), corner_color, corner_size)
        cv.line(frame, (x + w, y), (x + w - edge_len, y), corner_color, corner_size)
        cv.line(frame, (x, y + h), (x, y + h - edge_len), corner_color, corner_size)
        cv.line(frame, (x, y + h), (x + edge_len, y + h), corner_color, corner_size)
        cv.line(frame, (x + w, y + h), (x + w - edge_len, y + h), corner_color, corner_size)
        cv.line(frame, (x + w, y + h), (x + w, y + h - edge_len), corner_color, corner_size)
        
    def show_frame(self,name,frame,x=0,y=0):
        
        if self.status:
            # frm = cv.resize(frame, (800, 420), cv.INTER_AREA)
            cv.moveWindow(name, x, y)
            cv.imshow(name, frame)
            cv.waitKey(self.FPS_MS)
        cv.waitKey(self.FPS_MS)
        
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()
        cv.destroyAllWindows()
        print(' > 🛑 Stop Object Detector')
        
    def checkDir(self):
        if not os.path.exists('images'):
            print(" > 📁 Create Directory images")
            os.mkdir('images')




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp', help='RTSP Stream', default=os.environ.get('RTSP_STREAM'))
    parser.add_argument('--location', help='Location', default='test_cameras')
    args = parser.parse_args()
    return args

def checkArgs(args):
    print(" > 📁 Location : ",args.location)
    print(" > 📁 RTSP Stream : ",args.rtsp)
    if args.rtsp is None:
        print(" > ❌ RTSP_STREAM is None => export RTSP_STREAM=<rtsp://xxxxx/:xxx/<path>>")
        return False
    return True


def checkEnvironment():
    if os.environ.get('API_URL') is None:
        print(" > ❌ API_URL is None => export API_URL=<http://xxxxx>")
        return False
    
    return True
def main():
    
    args = parse_args()
    
    if checkEnvironment() is False:
        exit()
    
    if checkArgs(args) is False:
        exit()
        
    tcamera = ObjectDetector(source=args.rtsp,location=args.location)
    tcamera.checkDir()    
        
    while True:
        try:
            tcamera.detector_rider()
        except AttributeError:
            pass
        
if __name__ == '__main__':
    
    main()
    
    
    
    
    
    
    
    