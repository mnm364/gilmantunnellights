import cv2, sys, os, select, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage import color
from itertools import product
from collections import deque
import time

class Video:
	def __init__(self):
		self.capture = cv2.VideoCapture(0)
		self.frame_rate = 30

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

		self.frame_buffer = deque()
		self.frame_buffer_sample_count = 0
		self.frame_buffer_sampling_period = 2
		self.frame_buffer_capacity = 50
		self.global_variance = None

		self.tt = 10
		self.tw = self.frame_buffer_capacity / (self.camera.frame_rate * 1.0 / self.frame_buffer_sampling_period)
		self.c = 1

	@staticmethod
	def show(displays, shape):
		for name, frame in displays.items():
			cv2.imshow(name, imresize(frame, shape, interp="nearest"))

	@staticmethod
	def snapshot(frame):
		if heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
			return frame
		return None

	# Take average across image (reduce computation)
	@staticmethod
	def blur(image, blur_size=30):
		h, w, c = image.shape
		bs = min(blur_size, h, w)
		average = np.empty((h / bs, w / bs, c))
		for i in range(h / bs):
			for j in range(w / bs):
				average[i, j, :] = np.mean(image[i*bs:i*(bs+1), j*bs:j*(bs+1), :], axis=(0,1), dtype=np.uint16)
		return average

	def calibrate(self, frame):
		self.frame_buffer_sample_count += 1
		if self.frame_buffer_sample_count < self.frame_buffer_sampling_period:
			return
		self.frame_buffer.append(frame)
		self.frame_buffer_sample_count = 0

		if len(self.frame_buffer) < self.frame_buffer_capacity:
			return

		print 'CALIBRATING'

		np_frame_buffer = np.stack([f for f in self.frame_buffer], axis=0)
		frame_buffer_mean = np.mean(np_frame_buffer, axis=0)
		# print frame_buffer_mean
		# vv = np.sum((np_frame_buffer - frame_buffer_mean)**2, axis=0) / len(self.frame_buffer)
		frame_buffer_variance = np.var(np_frame_buffer, axis=0)
		print frame_buffer_variance
		# print 'VV', vv

		f = lambda v, t: \
			v + 1 if t < self.tt else \
			pow((float(t) - self.tt) / self.tw, self.c) * (v + 1) + v
		print 'A', np.mean(frame_buffer_variance) + 1
		do = False
		if self.global_variance is None:
			do = True
		# print 'B', f(np.mean(self.global_variance), time.time() - self.start_time)
		if do or np.mean(frame_buffer_variance) + 1 <= f(np.mean(self.global_variance), time.time() - self.start_time):
			self.background = Colorizer.blur(frame_buffer_mean)
			self.global_variance = frame_buffer_variance
			self.frame_buffer = deque() # empty queue
			self.start_time = time.time()
		else:
			self.frame_buffer.clear()

	def run(self, block_size=20):
		displays = {}
		self.background = None
		self.start_time = 0
		for frame in self.camera.stream():
			h, w, c = frame.shape

			Colorizer.show(displays, frame.shape)
			displays['raw'] = frame

			# Take background picture override
			snap = Colorizer.snapshot(frame)
			if snap is not None:
				self.start_time = time.time()
				self.background = Colorizer.blur(snap)
				# self.global_variance = np.var(self.background, axis=2)
				# self.global_variance = np.full(self.background.shape, 100.0, dtype=np.float)
				self.global_variance = None
				# NOTE: ^ this makes no sense and should be changed
			elif self.background is None:
				continue

			self.calibrate(frame)
			blurred = Colorizer.blur(frame)
			diff = Colorizer.difference(self.background, blurred)

			a = Colorizer.pattern_variance(diff, len(Colorizer.COLORS))
			b = 100
			if self.global_variance is not None:
				b = Colorizer.pattern_variance(self.global_variance, len(Colorizer.COLORS))

			if a+1 <= b:
				print 'SKIP', a, b
				black = np.zeros(frame.shape)
				displays['pattern'] = black
				displays['difference'] = black
				displays['discrete blur'] = black
				continue
			else:
				print 'DISP', a, b

			# difference = np.empty(blurred.shape, dtype=np.uint8)
			# for chan in range(c):
				# difference[:,:,chan] = np.abs(np.subtract(background[:,:,chan], average[:,:,chan]))


			pattern = self.patternize(diff, shape=frame.shape)
			displays['pattern'] = pattern
			displays['difference'] = diff
			displays['discrete blur'] = blurred
			# cv2.imshow('pattern', imresize(pattern, frame.shape, interp="nearest"))
			# cv2.imshow("background", imresize(background, frame.shape, interp="nearest"))
			# cv2.imshow("difference", imresize(difference, frame.shape, interp="nearest"))

			# cv2.imshow("frame", frame)
			# cv2.imshow("average", imresize(average, frame.shape, interp="nearest"))

			# # Take picture	
			# if Colorizer.heardEnter() or cv2.waitKey(1)&0xFF == ord('s'):
			# 	background = average

		self.camera.release()
		cv2.destroyAllWindows()

	@staticmethod
	def pattern_variance(image, split):
		# print image.shape
		# print image
		boxes = product(range(split), repeat=2)
		# variances = []
		# for box in boxes:
			# np.mean(np.var())
		step = image.shape[0] / split
		return np.mean(np.mean([np.var(image[step*b[0]:step*(b[0]+1), step*b[1]:step*(b[1]+1)]) for b in boxes]))

	@staticmethod
	def difference(a, b):
		diff = np.abs(np.subtract(a, b)) # NOTE: not sure why this doesn't work :'(
		# diff = np.empty(a.shape, dtype=np.uint8)
		# for c in range(a.shape[2]):
		# 	diff[:,:,c] = np.abs(np.subtract(a[:,:,c], b[:,:,c]))
		return diff

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

