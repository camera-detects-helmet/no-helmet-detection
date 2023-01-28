import base64
import json
import os
import time
import cv2
import requests
import torch
import uuid

class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)
    

"""
load model and set confidance score
"""
def load_model(model_path,conf=0.5):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model
    model.conf = conf
    return model

"""
load video from source
"""
def load_video(source=0):
    vid = cv2.VideoCapture(source)
    return vid

"""
rescale frame
"""
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

"""
show frame
"""
def show_frame(frame_name,frame,x,y):
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(frame_name, x, y)
    cv2.imshow(frame_name, frame)

def send_image(path_normal,path_rider):
    img_normal = cv2.imread(path_normal)
    img_rider = cv2.imread(path_rider)
    
    #encode image
    _,buffer_normal = cv2.imencode('.jpg',img_normal)
    _,buffer_rider = cv2.imencode('.jpg',img_rider)
    jpg_normal = base64.b64encode(buffer_normal)
    jpg_rider = base64.b64encode(buffer_rider)
    
    os.remove(path_normal)
    os.remove(path_rider)
    
    #send image
    url = "http://localhost:8000/car"
    payload = json.dumps({
        "location": location,
        "base64DefaultImg": jpg_normal,
        "base64RiderImg": jpg_rider
    }, cls=BytesEncoder)
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    response = requests.post(url, headers=headers, data=payload, timeout=5)
    
    print(response.status_code, response.text, )
    
"""
main function
"""
if __name__ == '__main__':
    
    # load model
    model_helmet = load_model('models/helmet_best.pt',0.5)
    model_rider = load_model('models/rider_best.pt',0.9)
    
    if not os.path.exists('images'):
        os.mkdir('images')
    
    # load video and set fps
    source = 0
    # source = "http://"
    vid = load_video(source)
    vid.set(cv2.CAP_PROP_FPS, 10)
    
    # check video is open
    if not vid.isOpened():
        print("Cannot open camera")
        exit()
    
    # set variable and fps limit 
    rider = 0
    no_helmet = 0
    fpsLimit = 5
    startTime = time.time()
    sleep_time = 1.0/fpsLimit
    crop_status = False
    location = "TestLab"
    # main loop
    while True:
        
        """
        get video fps and total frames
        """
        video_fps = vid.get(cv2.CAP_PROP_FPS),
        total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

        """
        read video and rescale frame and set time
        """
        ret,frame = vid.read()
        frame50 = rescale_frame(frame, percent=50)
        nowTime = time.time()

        """
        detect rider in frame and filter class rider by lambda function               
        """
        result_rider = model_rider(frame50)
        predictions_rider = result_rider.pandas().xyxy[0].values.tolist()
        lamda_rider = list(filter(lambda x: x[6] == 'rider', predictions_rider))
        rider = len(lamda_rider)
        
        """
        check class rider is not empty and draw rectangle on rider frame and 
        """
        if rider > 0:
            
            for i in lamda_rider:
                
                """
                crop rider frame and crop a top region of rider frame
                """
                crop_rider = frame50[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                top_xmin, top_ymin, top_xmax, top_ymax = int(i[0]), int(i[1]), int(i[2]), (int(i[1]) + int(i[3]))/2
                top_region = frame50[int(top_ymin):int(top_ymax), int(top_xmin):int(top_xmax)]
                
                """
                detect helmet in top region of rider frame
                and filter class no_helmet by lambda function
                """
                detect_helmet = model_helmet(top_region)
                predictions_helmet = detect_helmet.pandas().xyxy[0].values.tolist()
                lambda_no_helmet = list(filter(lambda x: x[6] == 'no_helmet', predictions_helmet))
                no_helmet = len(lambda_no_helmet)
                
                """
                check class no_helmet is not empty and draw rectangle on top region of rider frame and 
                crop head frame  and show head frame and top region of rider frame 
                """
                if no_helmet > 0:
                    crop_status = True
                    for k in lambda_no_helmet:
                        
                        """
                        crop head frame and draw rectangle on top region of rider frame
                        """
                        crop_head = top_region[int(k[1]):int(k[3]), int(k[0]):int(k[2])]
                        cv2.rectangle(top_region, (int(k[0]),int(k[1])), (int(k[2]),int(k[3])), (0,0,255), 2)
                        show_frame('Head Frame',crop_head,500,900)
                
                """
                show rider frame and top region of rider frame
                """
                show_frame('Rider Frame',crop_rider,10,600)
                show_frame('Top Region of rider frame',top_region,500,600)
            """
            draw rectangle on rider frame and show main frame 
            """
            cv2.rectangle(frame50, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), 2)
            show_frame('Main Frame',frame50,10,0)
            if crop_status == True:
                id = str(uuid.uuid4())
                normal_img = f'./images/{location}_NormalSize_{id}.jpg'
                rider_img = f'./images/{location}_RiderSize_{id}.jpg'
                cv2.imwrite(normal_img,frame)
                cv2.imwrite(rider_img,crop_rider)
                send_image(normal_img,rider_img)                
                time.sleep(3)
                crop_status = False
        else:
            show_frame('Main Frame',frame50,10,0)
            
        """
        print fpsLimit and TimeDiff and sleep time and show fps and total frames
        """
        
        time.sleep(sleep_time)
        print("fpsLimit : ", fpsLimit , "TimeDiff : ", nowTime - startTime)
        startTime = time.time()
        print(" > 🏍 Rider : " , rider, ", ⛑ no helmet : ", no_helmet , ", 📹 FPS ", video_fps, ", 📼 Total Frames : ", total_frames)

        """
        press q to exit
        """
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    """
    release video and destroy all windows
    """
    vid.release()
    cv2.destroyAllWindows()
    
    