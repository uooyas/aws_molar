import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import json
import mysql.connector
from datetime import datetime

# Updated MySQL connection settings
db_config = {
    'host': 'ugmni-mysql.cpe0a008coe5.ap-northeast-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'Aws00100!',
    'database': 'Refashion'
}

def init_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INT AUTO_INCREMENT PRIMARY KEY,
                  timestamp DATETIME,
                  image_name VARCHAR(255),
                  input_image LONGBLOB,
                  output_image LONGBLOB,
                  label VARCHAR(50),
                  confidence FLOAT,
                  bounding_box JSON,
                  dominant_color JSON)''')
    conn.commit()
    conn.close()

def insert_detection(image_name, input_image, output_image, label, confidence, bounding_box, dominant_color):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = """INSERT INTO detections 
               (timestamp, image_name, input_image, output_image, label, confidence, bounding_box, dominant_color) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    values = (datetime.now(), image_name, input_image, output_image, label, confidence, 
              json.dumps(bounding_box), json.dumps(dominant_color))
    cursor.execute(query, values)
    conn.commit()
    conn.close()

def process_image(image_path):
    # Object detection model loading (YOLO)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "yolo", "yolov3.weights")
    cfg_path = os.path.join(script_dir, "yolo", "yolov3.cfg")
    net = cv2.dnn.readNet(weights_path, cfg_path)

    # Load class names
    classes_path = os.path.join(script_dir, "yolo", "coco.names")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    height, width = image.shape[:2]

    # Object detection
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # Process detected objects
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process and save results
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    results = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = float(confidences[i])
            color = colors[class_ids[i]]
            
            # Extract object region
            object_region = image[y:y+h, x:x+w]
            
            # Extract color
            object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
            object_region_reshaped = object_region_rgb.reshape((-1, 3))
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(object_region_reshaped)
            
            # RGB values of the dominant color
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color.tolist(), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y + 20), font, 2, color.tolist(), 2)
            cv2.putText(image, f"Color: RGB{tuple(dominant_color)}", (x, y + 45), font, 1, color.tolist(), 2)
            
            bounding_box = {"x": x, "y": y, "w": w, "h": h}
            
            results.append({
                "label": label,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "dominant_color": dominant_color.tolist()
            })
    else:
        # No objects detected
        label = "No object detected"
        confidence = 0.0
        bounding_box = {"x": 0, "y": 0, "w": 0, "h": 0}
        dominant_color = [0, 0, 0]

    # Convert images to binary for DB storage
    _, input_image_binary = cv2.imencode('.png', cv2.imread(image_path))
    _, output_image_binary = cv2.imencode('.jpg', image)

    # Save results and images to DB
    image_name = os.path.basename(image_path)
    
    # Insert detection for each detected object or once if no object detected
    if results:
        for result in results:
            insert_detection(image_name, input_image_binary.tobytes(), output_image_binary.tobytes(), 
                             result['label'], result['confidence'], result['bounding_box'], result['dominant_color'])
    else:
        insert_detection(image_name, input_image_binary.tobytes(), output_image_binary.tobytes(), 
                         label, confidence, bounding_box, dominant_color)

    return results

# Usage example
if __name__ == "__main__":
    image_path = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/images/image.png"
    
    if not os.path.exists(image_path):
        print(f"Error: The image file does not exist at {image_path}")
    else:
        try:
            results = process_image(image_path)
            if results:
                print("Image processing completed. Results:")
                print(json.dumps(results, indent=2))
            else:
                print("No objects detected in the image.")
            
            print("\nStored detections in the database:")
            detections = get_detections()
            print(json.dumps(detections, indent=2))
            
            print(f"\nInput and processed images have been stored in the database.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")