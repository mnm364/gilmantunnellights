import cv2, sys, os, select
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage import color
from itertools import product
import argparse

"""
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

	amax = max(product(range(split_count), repeat=2), key=lambda i: mean(diff, i[0], i[1], column_size))

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
		cv2.imshow('average', imresize(average, frame.shape, interp="nearest"))

		# Take picture	
		if heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
			background = average

		# Exit capture
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()

"""

class Video:
	def __init__(self):
		self.capture = cv2.VideoCapture(0)

	def stream(self):
		while True:
			ret, frame = self.capture.read()
			if not ret:
				raise Exception("Invalid video capture source")

			yield frame			

			# Exit capture
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

class Colorizer:

	COLORS = [
		(255, 0, 0),	# red
		(255, 127, 0),	# orange
		(255, 255, 0),	# yellow
		(0, 255, 0),	# green
		(0, 0, 255),	# blue
		(75, 0, 130),	# indigo
		(139, 0, 255)	# violet
	]

	def __init__(self, pattern):
		self.camera = Video()

		self.pattern = None
		if pattern == "line":
			self.pattern = Colorizer.line
		elif pattern == "square":
			self.pattern = Colorizer.square

	def run(self, block_size=20):
		background = np.empty([])
		for frame in self.camera.stream():
			h, w, c = frame.shape
			block_size = min(block_size, h, w)

			# Take average across image (reduce computation)
			average = np.empty((h / block_size, w / block_size, c))
			for i in range(h / block_size):
				for j in range(w / block_size):
					bs = block_size
					average[i, j, :] = np.mean(frame[i*bs:i*(bs+1), j*bs:j*(bs+1), :], axis=(0,1))

			if background.size > 1:
				difference = np.empty(average.shape, dtype=np.uint8)
				for chan in range(c):
					difference[:,:,chan] = np.abs(np.subtract(background[:,:,chan], average[:,:,chan]))
				pattern = Colorizer.square(difference)
				cv2.imshow('pattern', imresize(pattern, frame.shape, interp="nearest"))
				cv2.imshow("background", imresize(background, frame.shape, interp="nearest"))
				cv2.imshow("difference", imresize(difference, frame.shape, interp="nearest"))

			cv2.imshow("frame", frame)
			cv2.imshow("average", imresize(average, frame.shape, interp="nearest"))

			# Take picture	
			if Colorizer.heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
				background = average

		self.camera.release()
		cv2.destroyAllWindows()

	@staticmethod
	def line(difference_image, num_cols=None):
		# pattern = np.empty(difference_image.shape, dtype=np.uint8)
		pass

	@staticmethod
	def square(difference_image, num_cols=None, num_rows=None):
		pattern = np.zeros(difference_image.shape, dtype=np.uint8)
		h, w, c = pattern.shape

		# Define interval size of pattern
		step = w / (num_cols if num_cols else len(Colorizer.COLORS))

		# Calculate square of max difference to place pattern
		squares = product(range(len(Colorizer.COLORS)), repeat=2) # generator of tuples (int, int)
		max_square = max(squares, key=lambda s: np.mean(difference_image[step*s[0]:step*(s[0]+1), step*s[1]:step*(s[1]+1)]))

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[sum(max_square) % len(Colorizer.COLORS)]

		# Broadcast color to square to make pattern
		pattern[step*max_square[0]:step*(max_square[0]+1), step*max_square[1]:step*(max_square[1]+1)] = color_pick

		return pattern

	@staticmethod
	def heardEnter():
		i,o,e = select.select([sys.stdin],[],[],0.0001)
		for s in i:
			if s == sys.stdin:
				input = sys.stdin.readline()
				return True
		return False


def get_args():
	parser = argparse.ArgumentParser(description="Colorize video based on changes.")

	parser.add_argument("--pattern", type=str, required=True, choices=["line", "square"], help="Pattern to colorize video.")
	parser.add_argument("--update-freq", type=float, required=False, default=20, help="Update background with this periodicty (s).")

	args = parser.parse_args()
	check_args(args)
	return args

def check_args(args):
	pass

def main():
	args = get_args()
	c = Colorizer(args.pattern)
	c.run()


if __name__ == '__main__':
	main()

