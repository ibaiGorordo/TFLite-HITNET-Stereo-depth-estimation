# TFLite-HITNET-Stereo-depth-estimation
Python scripts form performing stereo depth estimation using the HITNET model in Tensorflow Lite.

![Hitnet stereo depth estimation TFLite](https://github.com/ibaiGorordo/TFLite-HITNET-Stereo-depth-estimation/blob/main/docs/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * **OpenCV**, **imread-from-url** and **tensorflow==2.6.0 or tflite_runtime**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

For the tflite runtime, you can either use tensorflow(make sure it is version 2.6.0 or above) `pip install tensorflow==2.6.0` or the [TensorFlow Runtime binary](https://github.com/PINTO0309/TensorflowLite-bin)

# Known issues
In computers with a GPU, the program would silently creash without any error during the inference, `os.environ["CUDA_VISIBLE_DEVICES"]="-1"` is added at the beginning of the script to force the program to run on the CPU. You can comment this line for other types of devices.

# tflite model
The original models were converted to different formats (including .tflite) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET) and save them into the **[models](https://github.com/ibaiGorordo/TFLite-HITNET-Stereo-depth-estimation/tree/main/models)** folder. 

# Original Tensorflow model
The Tensorflow pretrained model was taken from the [original repository](https://github.com/google-research/google-research/tree/master/hitnet).
 
# Examples

 * **Image inference**:
 
 ```
 python imageDepthEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoDepthEstimation.py
 ```
 
 * **DrivingStereo dataset inference**:
 
 ```
 python drivingStereoTest.py
 ```
 

# Pytorch inference
For performing the inference in Tensorflow, check my other repository **[HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation)**.

# ONNX inference
For performing the inference in ONNX, check my other repository **[ONNX HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation)**.

# [Inference video Example Raspberry Pi 4](https://youtu.be/-6zhFs9X8Rg) 
 ![Hitnet stereo depth estimation on video Raspberry Pi 4](https://github.com/ibaiGorordo/TFLite-HITNET-Stereo-depth-estimation/blob/main/docs/img/Pi4tfliteHitnetDepthEstimation.gif)

# References:
* Hitnet model: https://github.com/google-research/google-research/tree/master/hitnet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: https://arxiv.org/abs/2007.12140
 
