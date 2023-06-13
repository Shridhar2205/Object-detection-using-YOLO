#image processing library
import cv2
#command-line argument parsing library
import argparse as a
#library for numerical operations
import numpy as np
import imutils
import time

#piece of code to parse command line arguments i.e., take input of the image, config file, weights file and class names file
ap = a.ArgumentParser()
'''ap.add_argument('-i', '--image', required=True, help = 'path to input image')'''
ap.add_argument('-i', '--input', required=True,	
                help='path to input video')
ap.add_argument('-o', '--output', required=True, 
                help='path to output video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


#read class names from text file and store them in a list
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#read pre-trained model and config file and create the network using them
net = cv2.dnn.readNet(args.weights, args.config)
scale = 0.00392

vs = cv2.VideoCapture(args.input)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(get_output_layers(net))
    end = time.time()   
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    #loop over each of the output layers
    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            #to find the index of the class with the maximum score
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #to consider only those detections that have a confidence greater than the threshold i.e., 0.5
            if confidence > 0.5:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
            
    #apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #function to draw a bounding box and label on the input image for a detected object
    def draw_prediction(frame, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(frame, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(frame, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                            
                draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))
    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
