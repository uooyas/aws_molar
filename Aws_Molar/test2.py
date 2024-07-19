import cv2
import numpy as np
import os

# 모델 파일 다운로드 함수
def download_model(url, filename):
    import urllib.request
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    return filename

# 모델 파일 경로 설정
model_path = download_model("https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pb", "ssd_mobilenet_v2_coco_2018_03_29.pb")
config_path = download_model("https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# COCO 클래스 이름
class_names = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
               "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate",
               "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
               "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
               "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
               "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

# 의류 관련 클래스 ID
clothing_class_ids = [1, 27, 28, 31, 32, 33]  # person, hat, backpack, shoe, handbag, tie

# 모델 로드
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# 색상 분석 함수
def get_dominant_color(image):
    pixels = cv2.resize(image, (1, 1)).reshape(-1, 3)
    return pixels[0]

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

def detect_clothing(image, confidence_threshold=0.5):
    height, width = image.shape[:2]
    
    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 객체 검출
    outputs = net.forward()
    
    detected_objects = []
    
    for detection in outputs[0, 0, :, :]:
        confidence = float(detection[2])
        class_id = int(detection[1])
        
        if confidence > confidence_threshold and class_id in clothing_class_ids:
            x = int(detection[3] * width)
            y = int(detection[4] * height)
            w = int(detection[5] * width) - x
            h = int(detection[6] * height) - y
            
            # 객체 영역 추출
            object_region = image[max(0, y):min(y+h, height), max(0, x):min(x+w, width)]
            
            if object_region.size > 0:  # 영역이 유효한 경우에만 처리
                # 주요 색상 추출
                dominant_color = get_dominant_color(object_region)
                color_name_text = color_name(dominant_color)
                
                detected_objects.append({
                    "object": class_names[class_id],
                    "color": color_name_text,
                    "confidence": f"{confidence:.2f}",
                    "box": (x, y, w, h)
                })
    
    return detected_objects

# 메인 함수
def main(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 의류 검출
    detected_clothes = detect_clothing(image)

    # 결과 출력
    print("감지된 의류:")
    for obj in detected_clothes:
        print(f"- 객체: {obj['object']}, 색상: {obj['color']}, 신뢰도: {obj['confidence']}")

    # 결과 시각화
    for obj in detected_clothes:
        x, y, w, h = obj['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{obj['object']}: {obj['color']}"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # 여기에 실제 이미지 경로를 입력하세요
    main(image_path)