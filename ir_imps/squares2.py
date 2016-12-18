import cv2, sys, os, select
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import itertools
from skimage import color
import argparse

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

	sqrs = np.empty(orginal.shape, dtype=np.uint8)
	partition_size = sqrs.shape[1] / len(colors)
	column_size = diff.shape[1] / len(colors)

	# argmax = max(range(len(colors)), key=lambda c: np.sum(diff[:,c*diff_partition_size:c*diff_partition_size+diff_partition_size]))
	# argmax = max(range(split_count), key=lambda i: np.sum(column(diff, i, column_size)))
	# for a in itertools.product(range(split_count), repeat=2):
		# print a

	amax = max(itertools.product(range(split_count), repeat=2), key=lambda i: mean(diff, i[0], i[1], column_size))

	psize = partition_size
	csize = column_size
	sqrs[psize*amax[0]:psize*(amax[0]+1), psize*amax[1]:psize*(amax[1]+1)] = colors[amax[0]]

	# lines[argmax*partition_size:argmax*partition_size+partition_size,argmax*partition_size:argmax*partition_size+partition_size] = colors[argmax]
	cv2.imshow('sqr', sqrs)

def column(image, index, step):
	return image[:, step*index:step*(index+1)]

def mean(image, i, j, step):
	return np.mean(image[step*i:step*(i+1), step*j:step*(j+1)])

def split_columns(num, image):
	cs = image.shape[1] / num # column size
	for i in range(num):
		yield image[:,cs*i:cs*(i+1)] 

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
		average = np.empty((height/bs, width/bs, channels))
		# for a in range(0, height-bs, bs):
			# for b in range(0, width-bs, bs):
		for a in range(height/bs):
			for b in range(width/bs):
				# print np.mean(frame[a:a+bs, b:b+bs,:], axis=(0,1)).shape
				# exit(1)
				average[a,b,:] = np.mean(frame[a*bs:a*(bs+1), b*bs:b*(bs+1),:], axis=(0,1))
				# for c in range(channels):
					# average[a,b,c] = frame[a*bs:a*(bs+1), b*bs:b*(bs+1),c].mean()

		# calculate difference matrix
		if background.size > 1:
			
			# Display the background (and display as reasonable size)
			cv2.imshow('background', imresize(background, float(bs), interp='nearest'))

			diff = np.empty(average.shape, dtype=np.uint8)
			for c in range(channels):
				# TODO - convert to different color space

				# lab_background = color.rgb2lab(background)
				# lab_average = color.rgb2lab(average)
				diff[:,:,c] = np.abs(np.subtract(background[:,:,c], average[:,:,c]))
				# diff[:,:,c] = np.linalg.norm(np.subtract(background[:,:,c], average[:,:,c]))
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




def get_args():
	parser = argparse.ArgumentParser(description="Colorize video based on changes.")

	parser.add_argument("--pattern", type=str, required=True, choices=["line", "square"], help="Pattern to colorize video.")
	parser.add_argument("--update-freq", type=float, required=False, default=20, help="Update background with this periodicty (s).")

	check_args(args)
	return None

def check_args():
	pass

def main():
	args = get_args()
	


if __name__ == '__main__':
	main()

