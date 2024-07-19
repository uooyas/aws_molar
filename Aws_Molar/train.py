import cv2
import numpy as np
from sklearn.cluster import KMeans

# 물체 탐지 모델 로드 (YOLO)
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

# 클래스 이름 로드
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 캡처
image = cv2.imread("C:/Users/parks/OneDrive/바탕 화면/Aws_Molar/images/image.png")
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

# 결과 처리 및 표시
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        
        # 물체 영역 추출
        object_region = image[y:y+h, x:x+w]
        
        # 색상 추출
        object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
        object_region_reshaped = object_region_rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(object_region_reshaped)
        
        # 주요 색상의 RGB 값
        dominant_color = kmeans.cluster_centers_[0]
        dominant_color = dominant_color.astype(int)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence}", (x, y + 20), font, 2, color, 2)
        cv2.putText(image, f"Color: RGB{tuple(dominant_color)}", (x, y + 45), font, 1, color, 2)
        
        print(f"Detected object: {label}, Confidence: {confidence}, Dominant color (RGB): {dominant_color}")

# 결과 이미지 표시
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지 저장
cv2.imwrite("output_image.jpg", image)