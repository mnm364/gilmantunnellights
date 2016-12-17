import numpy as np
import cv2
from matplotlib import pyplot as plt


def stream():

	#cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# create filters
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(frame,(5,5),10)

		# Display the resulting frame
		cv2.imshow('frame_raw', frame)
		cv2.imshow('frame_bw', gray)
		cv2.imshow('frame_blur', blur)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def main():
	stream()

if __name__ == '__main__':
	main()
