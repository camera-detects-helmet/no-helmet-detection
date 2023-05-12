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
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.cap.set(cv.CAP_PROP_AUTOFOCUS, 1) # turn the autofocus off
        # self.cap.set(28, 255)
        self.cap.set(cv.CAP_PROP_FPS, 60)
        

        #get open cv version
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        if int(major_ver)  < 3 :
            self.fps = self.cap.get(cv.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
        else :
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            print("Frames per second using video.get(cv.CAP_PROP_FPS) : {0}".format(self.fps)) 
            
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
    
    def get_fps(self):
        return self.fps
