import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
print("Current working directory:", os.getcwd())
BASE_DIR = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar"
YOLO_DIR = os.path.join(BASE_DIR, "yolo")

weights_path = os.path.join(YOLO_DIR, "yolov3.weights")
cfg_path = os.path.join(YOLO_DIR, "yolov3.cfg")

net = cv2.dnn.readNet(weights_path, cfg_path)
print(f"weights file exists: {os.path.exists(weights_path)}")
print(f"cfg file exists: {os.path.exists(cfg_path)}")
classes = []
with open(os.path.join(YOLO_DIR, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)

def color_name(color):
    r, g, b = color
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > max(g, b):
        return "Red"
    elif g > max(r, b):
        return "Green"
    elif b > max(r, g):
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    else:
        return "Unknown"

def process_image(image_path):
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            object_region = frame[y:y+h, x:x+w]
            dominant_color = get_dominant_color(object_region)
            color_name_text = color_name(dominant_color)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {color_name_text}", (x, y - 5), font, 1, (0, 255, 0), 2)
            
            print(f"Detected {label} with color {color_name_text}")

    output_path = os.path.join(os.path.dirname(image_path), f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, frame)
    print(f"Output image saved to {output_path}")

# 이미지 처리
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_DIR, filename)
        print(f"Processing {image_path}")
        process_image(image_path)

print("All images processed.")