import cv2
import sys
import numpy as np

def main():

    # input
    video_capture = cv2.VideoCapture(0)
    img = cv2.imread('jay.png', -1)

    # load in trained facial recognition models
    cascades = []
    for i in range(1, len(sys.argv)):
        cascades.append(cv2.CascadeClassifier(sys.argv[i]))

    while True:

        # grab frame from feed
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        faces = []
        for cascade in cascades:
            faces.append(detectfaces(cascade, gray))

        for face in faces:
            frame = overlay(frame, face, img)

        cv2.imshow('feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    video_capture.release()
    cv2.destroyAllWindows()

def overlay(frame, faces, img):
    for (x, y, w, h) in faces:
        print x,y,w,h
        scaled = cv2.resize(img, (w, h))
        for c in range(3):
            frame[y:y+h, x:x+w, c] = scaled[:,:,c] + frame[y:y+h, x:x+w, c] * (1.0 - scaled[:,:,3]/255.0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def detectfaces(model, frame):
    faces = model.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags = 0
        # flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT|cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
    )
    return faces

if __name__ == '__main__':
    main()
