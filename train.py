import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import json
import mysql.connector
from mysql.connector import pooling
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Create a connection pool
connection_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="mypool",
                                                              pool_size=5,
                                                              **db_config)

def get_db_connection():
    return connection_pool.get_connection()

def init_db():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                             (id INT AUTO_INCREMENT PRIMARY KEY,
                              timestamp DATETIME,
                              image_path VARCHAR(255),
                              label VARCHAR(50),
                              confidence FLOAT,
                              bounding_box JSON,
                              dominant_color JSON)''')
            conn.commit()
        logger.info("Database initialized successfully")
    except mysql.connector.Error as err:
        logger.error(f"Error initializing database: {err}")
        raise

def insert_detection(image_path, label, confidence, bounding_box, dominant_color):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                query = """INSERT INTO detections 
                           (timestamp, image_path, label, confidence, bounding_box, dominant_color) 
                           VALUES (%s, %s, %s, %s, %s, %s)"""
                values = (datetime.now(), image_path, label, confidence, 
                          json.dumps(bounding_box), json.dumps(dominant_color))
                cursor.execute(query, values)
            conn.commit()
        logger.info(f"Detection inserted for image: {image_path}")
    except mysql.connector.Error as err:
        logger.error(f"Error inserting detection: {err}")
        raise

def load_yolo_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "yolo", "yolov3.weights")
    cfg_path = os.path.join(script_dir, "yolo", "yolov3.cfg")
    
    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError("YOLO model files not found")
    
    net = cv2.dnn.readNet(weights_path, cfg_path)
    
    classes_path = os.path.join(script_dir, "yolo", "coco.names")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_objects(image, net, classes):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

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

    return boxes, confidences, class_ids

def get_dominant_color(image_region):
    image_region_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
    image_region_reshaped = image_region_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(image_region_reshaped)
    return kmeans.cluster_centers_[0].astype(int)

def process_image(image_path):
    try:
        net, classes = load_yolo_model()
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        boxes, confidences, class_ids = detect_objects(image, net, classes)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        results = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                
                object_region = image[y:y+h, x:x+w]
                dominant_color = get_dominant_color(object_region)
                
                cv2.rectangle(image, (x, y), (x + w, y + h), color.tolist(), 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y + 20), font, 2, color.tolist(), 2)
                cv2.putText(image, f"Color: RGB{tuple(dominant_color)}", (x, y + 45), font, 1, color.tolist(), 2)
                
                bounding_box = {"x": x, "y": y, "w": w, "h": h}
                
                insert_detection(image_path, label, confidence, bounding_box, dominant_color.tolist())

                results.append({
                    "label": label,
                    "confidence": confidence,
                    "bounding_box": bounding_box,
                    "dominant_color": dominant_color.tolist()
                })

        output_path = os.path.join(os.path.dirname(image_path), "output_image.jpg")
        cv2.imwrite(output_path, image)
        logger.info(f"Processed image saved to {output_path}")

        return results
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def get_detections():
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
                rows = cursor.fetchall()
        
        for row in rows:
            row['bounding_box'] = json.loads(row['bounding_box'])
            row['dominant_color'] = json.loads(row['dominant_color'])
            row['timestamp'] = row['timestamp'].isoformat()
        
        return rows
    except mysql.connector.Error as err:
        logger.error(f"Error fetching detections: {err}")
        raise

# Initial setup
if __name__ == "__main__":
    try:
        init_db()
        image_path = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/images/image.png"
        results = process_image(image_path)
        print(json.dumps(results, indent=2))
        
        detections = get_detections()
        print(json.dumps(detections, indent=2))
    except Exception as e:
        logger.error(f"An error occurred: {e}")