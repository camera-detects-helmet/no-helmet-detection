import cv2
import torch


def load_model(model_path,conf=0.5):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model
    model.conf = conf
    return model

def load_video():
    vid = cv2.VideoCapture(0)
    return vid

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

if __name__ == '__main__':
    
    model_helmet = load_model('models/helmet_best.pt',0.5)
    model_rider = load_model('models/rider_best.pt',0.9)
    vid = load_video()
    
    while(True):
        
        ret,frame = vid.read()
        frame50 = rescale_frame(frame, percent=50)
        result_rider = model_rider(frame50)
        predictions_rider = result_rider.pandas().xyxy[0].values.tolist()
        if len(predictions_rider) > 0:
            for i in predictions_rider:
                
                print("Rider : "+ str(i))
                
                crop_rider = frame50[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                detect_helmet = model_helmet(crop_rider)
                predictions_helmet = detect_helmet.pandas().xyxy[0].values.tolist()
                
                if len(predictions_helmet) > 0:
                
                    for k in predictions_helmet:
                        
                        print("Helmet : "+ str(k))
                        if k[6] == 'no_helmet':
                            crop_head = crop_rider[int(k[1]):int(k[3]), int(k[0]):int(k[2])]
                            cv2.rectangle(crop_rider, (int(k[0]),int(k[1])), (int(k[2]),int(k[3])), (0,0,255), 2)
                            cv2.imshow('crop head',crop_head)
                        else:
                            pass
                        
                cv2.imshow('crop rider',crop_rider)
            
            cv2.rectangle(frame50, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), 2)
            cv2.imshow('frame75',frame50)
        else:
            cv2.imshow('frame75',frame50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    
    
