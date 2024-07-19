import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# YOLO 모델 파일 경로 설정
yolo_weights = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/yolo/yolov3.weights"
yolo_config = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/yolo/yolov3.cfg"
coco_names = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/yolo/coco.names"
image_path = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/images/image.png"

# 파일 존재 여부 확인
for file_path in [yolo_weights, yolo_config, coco_names, image_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' 파일을 찾을 수 없습니다.")

# YOLO 모델 로드
net = cv2.dnn.readNet(yolo_weights, yolo_config)
classes = []
with open(coco_names, "r") as f:
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

# 이미지 로드
frame = cv2.imread(image_path)
height, width, channels = frame.shape

# YOLO 입력 준비
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # 객체 감지
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

detected_objects = []

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        
        # 객체 영역 추출
        object_region = frame[y:y+h, x:x+w]
        
        # 주요 색상 추출
        dominant_color = get_dominant_color(object_region)
        color_name_text = color_name(dominant_color)
        
        # 결과 저장
        detected_objects.append({
            "object": label,
            "color": color_name_text,
            "confidence": f"{confidence:.2f}"
        })

# 결과 출력
print("감지된 객체:")
for obj in detected_objects:
    print(f"- 객체: {obj['object']}, 색상: {obj['color']}, 신뢰도: {obj['confidence']}")

# 결과 이미지 저장 (GUI 표시 없이)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {detected_objects[i]['color']}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

cv2.imwrite("/home/ec2-user/aws_molar/aws_molar/Aws_Molar/images/result_image.png", frame)
print("결과 이미지가 저장되었습니다: /home/ec2-user/aws_molar/aws_molar/Aws_Molar/images/result_image.png")