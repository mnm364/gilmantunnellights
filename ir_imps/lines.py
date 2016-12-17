import cv2, sys, os, select
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

def heardEnter():
	i,o,e = select.select([sys.stdin],[],[],0.0001)
	for s in i:
		if s == sys.stdin:
			input = sys.stdin.readline()
			return True
	return False

def display_lines(diff, orginal, colors=None):
	if colors is None:
		colors = [
			(255, 0, 0),	# red
			(255, 127, 0),	# orange
			(255, 255, 0),	# yellow
			(0, 255, 0),	# green
			(0, 0, 255),	# blue
			(75, 0, 130),	# indigo
			(139, 0, 255)	# violet
		]

	lines = np.empty(orginal.shape, dtype=np.uint8)
	partition_size = lines.shape[1] / len(colors)
	diff_partition_size = diff.shape[1] / len(colors)

	argmax = max(range(len(colors)), key=lambda c: np.sum(diff[:,c*diff_partition_size:c*diff_partition_size+diff_partition_size]))

	lines[:,argmax*partition_size:argmax*partition_size+partition_size] = colors[argmax]
	cv2.imshow('lines', lines)

def main():
	
	background = np.empty([])
	plt.show()	
	capture = cv2.VideoCapture(0)
	while True:
		ret, frame = capture.read()
		height, width, channels = frame.shape
		
		bs = min(20, height, width)		

		# average pooling
		average = np.empty((height/bs,width/bs, 3))
		for a in range(height/bs):
			for b in range(width/bs):
				for c in range(channels):
					block = frame[a*bs:a*bs+bs, b*bs:b*bs+bs,c]
					average[a,b,c] = block.mean()	

		cv2.imshow('frame', frame)

		# calculate difference matrix
		if background.size > 1:
			cv2.imshow('background', imresize(background, 20.0))
			diff = np.empty(average.shape, dtype=np.uint8)
			for c in range(channels):
				diff[:,:,c] = np.abs(np.subtract(background[:,:,c], average[:,:,c]))
				plt.plot(diff[:,:,c].flatten())
				plt.draw()
				cv2.imshow('diff', imresize(diff, 20.0))
			display_lines(diff, frame)
	
		if heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
			background = average

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

