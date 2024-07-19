import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import json
import sqlite3
from datetime import datetime

# 절대 경로 설정
ROOT_DIR = r"C:\Users\parks\OneDrive\바탕 화면\Aws_Molar"
YOLO_DIR = os.path.join(ROOT_DIR, "yolo")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
DB_FILE = os.path.join(ROOT_DIR, 'detections.db')

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  image_path TEXT,
                  label TEXT,
                  confidence REAL,
                  bounding_box TEXT,
                  dominant_color TEXT)''')
    conn.commit()
    conn.close()

def insert_detection(image_path, label, confidence, bounding_box, dominant_color):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = """INSERT INTO detections 
               (timestamp, image_path, label, confidence, bounding_box, dominant_color) 
               VALUES (?, ?, ?, ?, ?, ?)"""
    values = (datetime.now().isoformat(), image_path, label, confidence, 
              json.dumps(bounding_box), json.dumps(dominant_color))
    cursor.execute(query, values)
    conn.commit()
    conn.close()

def process_image(image_path):
    # 물체 탐지 모델 로드 (YOLO)
    weights_path = os.path.join(YOLO_DIR, "yolov3.weights")
    cfg_path = os.path.join(YOLO_DIR, "yolov3.cfg")
    
    print(f"Weights path: {weights_path}")
    print(f"Config path: {cfg_path}")
    print(f"Do files exist? Weights: {os.path.exists(weights_path)}, Config: {os.path.exists(cfg_path)}")
    
    net = cv2.dnn.readNet(weights_path, cfg_path)

    # 클래스 이름 로드
    classes_path = os.path.join(YOLO_DIR, "coco.names")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    height, width = image.shape[:2]

    # 물체 탐지
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # 탐지된 물체 처리
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

    # 노이즈 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과 처리 및 저장
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    results = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = float(confidences[i])
            color = colors[class_ids[i]]
            
            # 물체 영역 추출
            object_region = image[y:y+h, x:x+w]
            
            # 색상 추출
            object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
            object_region_reshaped = object_region_rgb.reshape((-1, 3))
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(object_region_reshaped)
            
            # 주요 색상의 RGB 값
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color.tolist(), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y + 20), font, 2, color.tolist(), 2)
            cv2.putText(image, f"Color: RGB{tuple(dominant_color)}", (x, y + 45), font, 1, color.tolist(), 2)
            
            bounding_box = {"x": x, "y": y, "w": w, "h": h}
            
            # DB에 결과 저장
            insert_detection(image_path, label, confidence, bounding_box, dominant_color.tolist())

            results.append({
                "label": label,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "dominant_color": dominant_color.tolist()
            })

    # 결과 이미지 저장
    output_path = os.path.join(os.path.dirname(image_path), "output_image.png")
    cv2.imwrite(output_path, image)

    return results

# 데이터베이스에서 탐지 결과 조회
def get_detections():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# 초기 설정
init_db()

# 사용 예시
if __name__ == "__main__":
    image_path = os.path.join(IMAGE_DIR, "image.png")
    print(f"Image path: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
    results = process_image(image_path)
    print(json.dumps(results, indent=2))
    
    # 데이터베이스에서 결과 조회
    detections = get_detections()
    print(json.dumps(detections, indent=2))