from threading import Thread
import cv2 as cv
import numpy as np
import time
import os


RTSP_STREAM = "rtsp://localhost:8554/mystream"

class GetVideo:
    
    def __init__(self,stream):
        self.stream = stream
        self.cap = cv.VideoCapture(self.stream)
        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        while True:
            if self.stopped:
                return
            ret, self.frame = self.cap.read()
            if not ret:
                self.stop()
                return
            
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
        cv.destroyAllWindows()
        print("Video Stopped")
        
    def __del__(self):
        self.stop()
        
        
        
        
        
        
if __name__ == "__main__":
    video = GetVideo(RTSP_STREAM)
    while True:
        frame = video.read()
        if frame is not None:
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                video.stop()
                break
        else:
            print("No Frame")
            time.sleep(1)
            continue
        
        
    
    
    
        
        