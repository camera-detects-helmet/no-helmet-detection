from threading import Thread
import cv2 as cv


class LoadSourceFile:
    
    def __init__(self, source):
        
        self.source = source
        self.cap = cv.VideoCapture(self.source)
        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        self.cap.set(cv.CAP_PROP_FPS, 60)
        
        # self.cap.set(28, 255) 

        
    
    def config(self, width=640, height=480):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        
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
    
    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
        cv.destroyAllWindows()
        print("Video Stopped")
        
    def __del__(self):
        self.stop()
        
    def resize(self, frame, width=640, height=480):
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        
        

        
        

    
    