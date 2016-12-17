import cv2, sys, os, select
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import itertools

def heardEnter():
	i,o,e = select.select([sys.stdin],[],[],0.0001)
	for s in i:
		if s == sys.stdin:
			input = sys.stdin.readline()
			return True
	return False

def display_lines(diff, orginal, colors=None, split_count=None):
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
	if not split_count:
		split_count = len(colors)

	lines = np.empty(orginal.shape, dtype=np.uint8)
	partition_size = lines.shape[1] / len(colors)
	column_size = diff.shape[1] / len(colors)

	# argmax = max(range(len(colors)), key=lambda c: np.sum(diff[:,c*diff_partition_size:c*diff_partition_size+diff_partition_size]))
	argmax = max(range(split_count), key=lambda i: np.sum(column(diff, i, column_size))
	argmax = max(itertools.product(range(split_count)), key=lambda i,j: np.sum(square(diff, i, j, column_size)))

	lines[argmax*partition_size:argmax*partition_size+partition_size,argmax*partition_size:argmax*partition_size+partition_size] = colors[argmax]
	cv2.imshow('lines', lines)

def column(image, index, step):
	return image[:, step*index:step*(index+1)]

def square(image, i, j, step):
	return image[step*i:step*(i+1), step*j:step*(j+1)]	

def split_columns(num, image):
	cs = image.shape[1] / num # column size
	for i in range(num):
		yield image[:,cs*i:cs*(i+1)] 

def split_squares(num

def main():
	
	background = np.empty([])
	plt.show()	
	capture = cv2.VideoCapture(0)
	while True:
		ret, frame = capture.read()
		height, width, channels = frame.shape
	
		# block size for average pooling	
		bs = min(20, height, width)		

		# average pooling
		average = np.empty((height/bs,width/bs, 3))
		for a in range(0, height, bs):
			for b in range(0, width, bs):
				average[a,b,c] = np.mean(frame[a:a+bs, b:b+bs,:], axis=2)
#				for c in range(channels):
#					average[a,b,c] = frame[a:a+bs, b:b+bs,c].mean()



		# calculate difference matrix
		if background.size > 1:
			
			# Display the background (and display as reasonable size)
			cv2.imshow('background', imresize(background, float(bs), interp='nearest'))

			diff = np.empty(average.shape, dtype=np.uint8)
			for c in range(channels):
				# TODO - convert to different color space

				diff[:,:,c] = np.abs(np.subtract(background[:,:,c], average[:,:,c]))
				plt.plot(diff[:,:,c].flatten())
				plt.draw()
				cv2.imshow('diff', imresize(diff, 20.0))
			display_lines(diff, frame)

		# Display raw camera feed
		cv2.imshow('frame', frame)

		# Take picture	
		if heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
			background = average

		# Exit capture
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

