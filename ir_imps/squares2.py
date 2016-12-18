import cv2, sys, os, select, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage import color
from itertools import product

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

		# Select pattern algorithm
		self.patternize = None
		if pattern == "line":
			self.patternize = self.line
		elif pattern == "square":
			self.patternize = self.square
		elif pattern == "crosshair":
			self.patternize = self.crosshair

		self.state = None # (positionx, positiony, velocityx, velocityy)

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
				pattern = self.patternize(difference, shape=frame.shape)
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

	def line(self, difference_image, num_cols=None, shape=None):
		num_cols = num_cols if num_cols else len(Colorizer.COLORS)
		pattern = np.zeros(difference_image.shape, dtype=np.uint8)
		h, w, c = pattern.shape

		# Define interval size of pattern
		step = w / (num_cols if num_cols else len(Colorizer.COLORS))

		# Calculate column of max difference to place pattern
		max_col = max(range(num_cols), key=lambda s: np.mean(difference_image[:, step*s:step*(s+1)]))

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[max_col % len(Colorizer.COLORS)]

		# Broadcast color to line to make pattern
		pattern[:, step*max_col:step*(max_col+1)] = color_pick

		return pattern

	def square(self, difference_image, num_cols=None, num_rows=None, shape=None):
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

	def square2(self, difference_image, num_cols=None, num_rows=None, shape=None):
		num_cols = num_cols if num_cols else len(Colorizer.COLORS)
		pattern = np.zeros(difference_image.shape, dtype=np.uint8)
		h, w, c = pattern.shape

		# Define interval size of pattern
		step = w / num_cols

		# Calculate square of max difference to place pattern
		squares = product(range(num_cols), repeat=2) # generator of tuples (int, int)
		max_square = max(squares, key=lambda s: np.mean(difference_image[step*s[0]:step*(s[0]+1), step*s[1]:step*(s[1]+1)]))

		# set position and velocity
		if self.state is None:
			self.state = (max_square, (0, 0))

		# Calculate new position w/physics based dampening
		new_position = max_square
		if not Colorizer.at_edge(max_square, (num_cols, num_cols)):
			new_position = self.dampen(max_square)
		else:
			self.state = None

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[0]

		# Broadcast color to square to make pattern
		pattern[step*new_position[0]:step*(new_position[0]+1), step*new_position[1]:step*(new_position[1]+1)] = color_pick

		return pattern

	def crosshair(self, difference_image, num_cols=None, num_rows=None, shape=None):
		# NOTE: hack, shape must be defined
		if shape is None:
			raise Exception("internal error: hack for shape in crosshair, check failed.")

		num_cols = num_cols if num_cols else len(Colorizer.COLORS)
		num_rows = num_cols # FIXME
		pattern = np.zeros(shape, dtype=np.uint8)
		h, w, c = difference_image.shape

		# Define interval size of pattern
		step = w / num_cols

		# Calculate square of max difference to place pattern
		squares = product(range(num_cols), repeat=2) # generator of tuples (int, int)
		max_square = max(squares, key=lambda s: np.mean(difference_image[step*s[0]:step*(s[0]+1), step*s[1]:step*(s[1]+1)]))

		# Uniquely pick color for valid list
		color_pick = Colorizer.COLORS[0]

		boundary = 5 # TODO - dont hardcode magic numbers
		h2, w2, c = pattern.shape
		step_x2 = w2 / num_cols
		# step_y2 = h2 / num_rows
		step_y2 = step_x2
		for i in range(num_cols + 1):
			pattern[step_x2*max_square[0]+boundary:step_x2*(max_square[0]+1)-boundary, step_x2*i+boundary:step_x2*(i+1)-boundary] = color_pick
		for i in range(num_rows + 1):
			pattern[step_y2*i+boundary:step_y2*(i+1)-boundary, step_y2*max_square[1]+boundary:step_y2*(max_square[1]+1)-boundary] = color_pick

		# Broadcast color to square to make pattern
		# pattern[step*max_square[0]:step*(max_square[0]+1), step*max_square[1]:step*(max_square[1]+1)] = color_pick

		return pattern

	@staticmethod
	def at_edge(position, shape):
		if len(position) != len(shape):
			raise Exception("internal error: at edge position and shape must have same length")
		for i in range(len(shape)):
			if int(position[i]) == 0 or int(position[i]) == shape[i]:
				return True
		return False

	def dampen(self, position, delta_t=0.5, acceleration=-2):
		p0, v0 = self.state
		p = position
		print 'p0, v0', p0, v0
		print 'p', p
		vx = v0[0] * delta_t + 2 * acceleration * (p[0] - p0[0])
		vy = v0[1] * delta_t + 2 * acceleration * (p[1] - p0[1])
		print 'vx, vy', vx, vy
		xt = (vx + v0[0]) * delta_t + p0[0]
		yt = (vy + v0[1]) * delta_t + p0[1]
		self.state = (xt, yt), (vx, vy)
		print 'state', self.state
		print '-------------'
		# exit(1)
		return self.state[0][0], self.state[0][1]

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

	parser.add_argument("--pattern", type=str, required=True, choices=["line", "square", "crosshair"], help="Pattern to colorize video.")
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

