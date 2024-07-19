import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# YOLO 모델 파일 경로 설정 (전체 경로 사용)
yolo_weights = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/yolov3.weights"
yolo_config = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/yolov3.cfg"
coco_names = "/home/ec2-user/aws_molar/aws_molar/Aws_Molar/coco.names"

# 파일 존재 여부 확인
if not os.path.exists(yolo_weights):
    raise FileNotFoundError(f"'{yolo_weights}' 파일을 찾을 수 없습니다.")
if not os.path.exists(yolo_config):
    raise FileNotFoundError(f"'{yolo_config}' 파일을 찾을 수 없습니다.")
if not os.path.exists(coco_names):
    raise FileNotFoundError(f"'{coco_names}' 파일을 찾을 수 없습니다.")

# YOLO 모델 로드
net = cv2.dnn.readNet(yolo_weights, yolo_config)
classes = []
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 나머지 코드는 그대로 유지...

def get_dominant_color(image):
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 2D 배열로 재구성
    pixels = image.reshape((-1, 3))
    
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    
    # 주요 색상 추출
    dominant_color = kmeans.cluster_centers_[0]
    
    # 색상을 정수로 변환
    dominant_color = dominant_color.astype(int)
    
    return dominant_color

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

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            # 객체 영역 추출
            object_region = frame[y:y+h, x:x+w]
            
            # 주요 색상 추출
            dominant_color = get_dominant_color(object_region)
            color_name_text = color_name(dominant_color)
            
            # 결과 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {color_name_text}", (x, y - 5), font, 1, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow("Image", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()