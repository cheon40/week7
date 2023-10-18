from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(file):
    # 이미지를 업로드 폴더에 저장
    image_filename = str(np.random.randint(1, 100000)) + ".png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file.save(image_path)

    # 이미지 분석 결과 가져오기
    result = perform_image_analysis(image_path)

    # 이미지에 결과 텍스트 추가
    img = cv2.imread(image_path)

    # 사용할 폰트 지정 (예: cv2.FONT_HERSHEY_SIMPLEX)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 결과 텍스트 추가
    img = cv2.putText(img, result, (10, 30), font, 1, (0, 255, 0), 2)

    # 수정된 이미지 저장
    cv2.imwrite(image_path, img)

    return image_path, result

import cv2
import numpy as np

def perform_image_analysis(image_path):
    # YOLOv4 모델 로드
    net = cv2.dnn.readNet('yolo_weights/yolov4.cfg', 'yolo_weights/yolov4.weights')

    # 클래스 이름 로드
    with open('yolo_weights/coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # 이미지 읽기
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 이미지 전처리 및 YOLOv4에 전달
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # 결과 처리
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # 임계값 설정
                # 객체의 중심 좌표 및 너비, 높이 계산
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                # 객체의 좌표 계산
                x = int(center_x - obj_width / 2)
                y = int(center_y - obj_height / 2)

                # 이미지에 사각형 그리기
                cv2.rectangle(img, (x, y), (x + obj_width, y + obj_height), (0, 255, 0), 2)

                # 클래스 이름과 신뢰도를 텍스트로 추가
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 분석 결과 이미지 저장 (이 부분은 선택사항)
    result_image_path = 'static/uploads/annotated_image.png'
    cv2.imwrite(result_image_path, img)

    return result_image_path


def encode_image(image_path):
    # 이미지를 Base64로 인코딩
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 업로드된 이미지 처리
    file = request.files['file']
    if file:
        # 이미지 처리 함수 호출
        image_path, result = process_image(file)

        # 결과 및 이미지 경로를 HTML로 전달
        encoded_image = encode_image(image_path)
        return render_template('index.html', result=result, encoded_image=encoded_image)

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
