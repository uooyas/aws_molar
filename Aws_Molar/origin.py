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
                  image_path VARCHAR(255),
                  label VARCHAR(50),
                  confidence FLOAT,
                  bounding_box JSON,
                  dominant_color JSON)''')
    conn.commit()
    conn.close()

def insert_detection(image_path, label, confidence, bounding_box, dominant_color):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = """INSERT INTO detections 
               (timestamp, image_path, label, confidence, bounding_box, dominant_color) 
               VALUES (%s, %s, %s, %s, %s, %s)"""
    values = (datetime.now(), image_path, label, confidence, 
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
            
            # Save results to DB
            insert_detection(image_path, label, confidence, bounding_box, dominant_color.tolist())

            results.append({
                "label": label,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "dominant_color": dominant_color.tolist()
            })

    # Save result image
    output_path = os.path.join(os.path.dirname(image_path), "output_image.jpg")
    cv2.imwrite(output_path, image)

    return results

# Query detection results from the database
def get_detections():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        row['bounding_box'] = json.loads(row['bounding_box'])
        row['dominant_color'] = json.loads(row['dominant_color'])
        row['timestamp'] = row['timestamp'].isoformat()
    
    return rows

# Initial setup
init_db()

# Usage example
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    results = process_image(image_path)
    print(json.dumps(results, indent=2))
    
    # Query results from the database
    detections = get_detections()
    print(json.dumps(detections, indent=2))