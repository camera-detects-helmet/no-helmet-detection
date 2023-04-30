from threading import Thread
import cv2 as cv
import numpy as np
import time
import os
import argparse



class GetVideo:
    
    def __init__(self,rtsp_url):
        self.stream = rtsp_url
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
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp', type=str, default='', help='RTSP Stream URL')
    args = parser.parse_args()
    return args

def main():
    args = parse_opt()
    video = GetVideo(args.rtsp)
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
        

if __name__ == "__main__":    
    main()
        
    
    
    
        
        