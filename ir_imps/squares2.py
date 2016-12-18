import cv2, sys, os, select
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage import color
from itertools import product
import argparse

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
			# NOTE: ^ this just doesn't work...

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

		self.patternize = None
		if pattern == "line":
			self.patternize = Colorizer.line
		elif pattern == "square":
			self.patternize = Colorizer.square

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
				pattern = self.patternize(difference)
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
		num_cols = num_cols if num_cols else len(Colorizer.COLORS)
		pattern = np.zeros(difference_image.shape, dtype=np.uint8)
		h, w, c = pattern.shape

		# Define interval size of pattern
		step = w / (num_cols if num_cols else len(Colorizer.COLORS))

		# Calculate square of max difference to place pattern
		max_col = max(range(num_cols), key=lambda s: np.mean(difference_image[:, step*s:step*(s+1)]))

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[max_col % len(Colorizer.COLORS)]

		# Broadcast color to square to make pattern
		pattern[:, step*max_col:step*(max_col+1)] = color_pick

		return pattern

	@staticmethod
	def square(difference_image, num_cols=None, num_rows=None):
		num_cols = num_cols if num_cols else len(Colorizer.COLORS)
		pattern = np.zeros(difference_image.shape, dtype=np.uint8)
		h, w, c = pattern.shape

		# Define interval size of pattern
		step = w / num_cols

		# Calculate square of max difference to place pattern
		squares = product(range(num_cols), repeat=2) # generator of tuples (int, int)
		max_square = max(squares, key=lambda s: np.mean(difference_image[step*s[0]:step*(s[0]+1), step*s[1]:step*(s[1]+1)]))

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[sum(max_square) % len(Colorizer.COLORS)]

		# Broadcast color to square to make pattern
		pattern[step*max_square[0]:step*(max_square[0]+1), step*max_square[1]:step*(max_square[1]+1)] = color_pick

		return pattern

	@staticmethod
	def crosshairs(difference_image, num_cols=None, num_rows=None):
		pass

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

