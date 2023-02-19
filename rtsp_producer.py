# #brew install ffmpeg

# import cv2
# import subprocess as sp

# rtsp_server = 'rtsp://127.0.0.1:8554/mystream'
    
# cap = cv2.VideoCapture(0)
# sizeStr = str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
    
# command = ['ffmpeg',
#             '-re',
#             '-s', sizeStr,
#             '-r', str(fps),  # rtsp fps (from input server)
#             '-i', '-',
               
#             # You can change ffmpeg parameter after this item.
#             '-pix_fmt', 'yuv420p',
#             '-r', '30',  # output fps
#             '-g', '50',
#             '-c:v', 'libx264',
#             '-b:v', '2M',
#             '-bufsize', '64M',
#             '-maxrate', "4M",
#             '-preset', 'veryfast',
#             '-rtsp_transport', 'tcp',
#             '-segment_times', '5',
#             '-f', 'rtsp',
#             rtsp_server]

# process = sp.Popen(command, stdin=sp.PIPE)

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     ret2, frame2 = cv2.imencode('.png', frame)
#     process.stdin.write(frame2.tobytes())

import cv2 as cv
import subprocess as sp
from threading import Thread

rtsp_server = 'rtsp://127.0.0.1:8554/mystream'

class StreamingVideo:
    
    def __init__(self,src=0):
        self.cap = cv.VideoCapture(src)
        self.sizeStr = str(int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.command = ['ffmpeg',
                    '-re',
                    '-s', self.sizeStr,
                    '-r', str(self.fps),  # rtsp fps (from input server)
                    '-i', '-',
                       
                    # You can change ffmpeg parameter after this item.
                    '-pix_fmt', 'yuv420p',
                    '-r', '25',  # output fps
                    '-g', '50',
                    '-c:v', 'libx264',
                    '-b:v', '2M',
                    '-bufsize', '64M',
                    '-maxrate', "4M",
                    '-preset', 'veryfast',
                    '-rtsp_transport', 'tcp',
                    '-segment_times', '5',
                    '-f', 'rtsp',
                    rtsp_server]
        self.process = sp.Popen(self.command, stdin=sp.PIPE)
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            ret2, frame2 = cv.imencode('.png', frame)
            self.process.stdin.write(frame2.tobytes())
            
    def stop(self):
        self.process.terminate()
        self.cap.release()
        cv.destroyAllWindows()
        print("Video Stopped")
    
    def __del__(self):
        self.stop()
        
if __name__ == "__main__":
    video = StreamingVideo()
    
# brew install ffmpeg

