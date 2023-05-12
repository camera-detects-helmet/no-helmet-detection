# import the opencv library
import cv2
import numpy as np



  
# define a video capture object
vid = cv2.VideoCapture('01.mp4')

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv2.CAP_PROP_FPS, 10)
vid.set(cv2.CAP_PROP_AUTOFOCUS, 0) # 

delay = int(1000 / 10)

while(True):
    ret, frame = vid.read()

    
    
    if frame is not None:
        
        # Display the resulting frame
        cv2.imshow('Original', frame)
      
        print('Resolution: ' + str(frame.shape))
        print('FPS: ' + str(vid.get(cv2.CAP_PROP_FPS)))
    
    ret , frame = vid.read()
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

