import cv2
import pafy
import numpy as np
import glob
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img

# TODO: The app crashes when running on the gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

# out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (881*3,400))

# Get image list
left_images = glob.glob('DrivingStereo images/left/*.jpg')
left_images.sort()
right_images = glob.glob('DrivingStereo images/right/*.jpg')
right_images.sort()
depth_images = glob.glob('DrivingStereo images/depth/*.png')
depth_images.sort()

# Select model type
# model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400/saved_model_256x256/model_float32.tflite"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl/saved_model_256x256/model_float32.tflite"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d/saved_model_256x256/model_float32.tflite"

input_width = 256
camera_config = CameraConfig(0.546, 2000/1920*input_width) # rough estimate from the original calibration
max_distance = 30

# Initialize model
hitnet_depth = HitNet(model_path, model_type, camera_config)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images, right_images, depth_images):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/256

	# Estimate the depth
	disparity_map = hitnet_depth(left_img, right_img)
	depth_map = hitnet_depth.get_depth()

	color_disparity = draw_disparity(disparity_map)
	color_depth = draw_depth(depth_map, max_distance)
	color_real_depth = draw_depth(depth_img, max_distance)

	color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
	cobined_image = np.hstack((left_img,color_real_depth, color_depth))

	# out.write(cobined_image)
	cv2.imshow("Estimated depth", cobined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

# out.release()
cv2.destroyAllWindows()