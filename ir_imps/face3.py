import cv2
import sys
import numpy as np

cascPath = sys.argv[1]
# cascPath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascPath)
# profileCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

img = cv2.imread('jay.png', -1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # cv2.imshow('full', img)
    # cv2.waitKey(0)
    # cv2.imshow('scale', cv2.resize(img, (50, 50)))
    # cv2.waitKey(0)
    height, width, channels = frame.shape
    print height, width, channels
	rand = np.zeros((height,width,3), np.uint8)
    cv2.imshow('black', rand)
    continue

    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags = 0
    )

    scaled = None

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print x,y,w,h
        scaled = cv2.resize(img, (w, h))
        for c in range(3):
            frame[y:y+h, x:x+w, c] = scaled[:,:,c] + frame[y:y+h, x:x+w, c] * (1.0 - scaled[:,:,3]/255.0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# frame = cv2.resize(frame, (0,0), fx=2.0, fy=2.0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

