import time
import cv2
import numpy as np
from hitnet.utils_hitnet import *

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

drivingStereo_config = CameraConfig(0.546, 1000)

class HitNet():

	def __init__(self, model_path, model_type=ModelType.eth3d, camera_config=drivingStereo_config):

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0
		self.camera_config = camera_config

		# Initialize model
		self.model = self.initialize_model(model_path, model_type)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	def initialize_model(self, model_path, model_type):

		self.model_type = model_type

		self.interpreter = Interpreter(model_path=model_path, num_threads=4)
		self.interpreter.allocate_tensors()

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()

	def estimate_disparity(self, left_img, right_img):

		input_tensor = self.prepare_input(left_img, right_img)

		# Perform inference on the image
		if self.model_type == ModelType.flyingthings:
			left_disparity, right_disparity = self.inference(input_tensor)
			self.disparity_map = left_disparity
		else:
			self.disparity_map = self.inference(input_tensor)

		return self.disparity_map

	def get_depth(self):
		return self.camera_config.f*self.camera_config.baseline/self.disparity_map

	def prepare_input(self, left_img, right_img):

		left_img = cv2.resize(left_img, (self.input_width, self.input_height))
		right_img = cv2.resize(right_img, (self.input_width, self.input_height))

		if (self.model_type == ModelType.eth3d):

			# Shape (1, None, None, 2)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

			left_img = np.expand_dims(left_img,2)
			right_img = np.expand_dims(right_img,2)
			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
		else:
			# Shape (1, None, None, 6)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0

		return np.expand_dims(combined_img, 0).astype(np.float32)

	def inference(self, input_tensor):

		self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
		self.interpreter.invoke()
		left_disparity = self.interpreter.get_tensor(self.output_details[0]['index'])

		if self.model_type is not ModelType.flyingthings:
			return np.squeeze(left_disparity)

		right_disparity = self.interpreter.get_tensor(self.output_details[1]['index'])

		return np.squeeze(left_disparity), np.squeeze(right_disparity)

	def getModel_input_details(self):

		self.input_details = self.interpreter.get_input_details()
		input_shape = self.input_details[0]['shape']
		self.input_height = input_shape[1]
		self.input_width = input_shape[2]
		self.channels = input_shape[3]

	def getModel_output_details(self):

		self.output_details = self.interpreter.get_output_details()
		output_shape = self.output_details[0]['shape']


	






