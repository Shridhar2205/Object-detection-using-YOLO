# Object detection using deep learning with OpenCV and Python 
An implementation of object detection using the YOLO (You Only Look Once) algorithm. 
It uses the OpenCV library and the Darknet framework to perform the object detection.

Used a pre-trained weights file which has trained the YOLO model on a large dataset.
Used a configuration file which contains the model architecture and the hyperparameters required to initialise the YOLO network.
The YOLO model is capable of detecting upto 80 objects in an image, based on the classes set in the class(text) file.

To run the model for images, use the following command on any terminal:
python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

To download the weights file, go to this link:
https://pjreddie.com/media/files/yolov3.weights
Sources: https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9


UPDATE: Extended the algorithm to detect objects in a video.
To run the model for videos, use the following command on any terminal:
python yolo_opencv.py --input sample.mp4 --output sample_result.mp4 --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
