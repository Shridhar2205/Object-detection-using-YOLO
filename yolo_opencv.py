#image processing library
import cv2
#command-line argument parsing library
import argparse as a
#library for numerical operations
import numpy as np

#piece of code to parse command line arguments i.e., take input of the image, config file, weights file and class names file
ap = a.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

#function that takes YOLO network as input and returns the names of the output layers of the network
#uses getUnconnectedOutLayers() function to get names of unconnected output layers and then maps them to the actual layer names
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

#function to draw a bounding box and label on the input image for a detected object
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#read input image and get its width and height
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

#read class names from text file and store them in a list
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#read pre-trained model and config file and create the network using them
net = cv2.dnn.readNet(args.weights, args.config)

#generate a blob from the image to pass it to the network, and set the blob as the input to the network
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

#run forward pass through the network to get output of the output layers
outs = net.forward(get_output_layers(net))

#initialize lists to store detected bounding boxes, confidences and class IDs
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

#loop over each of the output layers
for out in outs:
    for detection in out:
        scores = detection[5:]
        #to find the index of the class with the maximum score
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #to consider only those detections that have a confidence greater than the threshold i.e., 0.5
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

#apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

#draw the bounding boxes and labels on the image
for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

#display output image
cv2.imshow("object detection", image)
cv2.waitKey()

#save output image
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
