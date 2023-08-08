from flask import Flask, render_template, Response
import cv2
import os
import time

app = Flask(__name__)

capture_start_time = None
camera = None  # 웹캠 객체를 전역 변수로 선언

# 웹캠으로부터 프레임을 가져오는 함수
def get_frame():
    global camera  # 전역 변수 사용

    if camera is None:
        camera = cv2.VideoCapture(0) #첫번째 카메라를 객체로 가져옴

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 프레임을 JPEG 이미지로 인코딩하여 반환
            ret, buffer = cv2.imencode('.jpg', frame) #카메라로부터 현재의 영상 읽어오기
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # camera.release()  # 웹캠 자원 해제 - 주석 처리


@app.route('/')  # app.route로 URL과 Flask 코드를 매핑해줌
def index():
    return render_template('index.html')


@app.route('/video_feed')  # 웹 브라우저에서 웹캠 스트리밍을 수신할 때 사용
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global camera  # 전역 변수 사용

    success, frame = camera.read()

    if success:
        image_name = f'captured_image.jpg'
        # 이미지 저장
        cv2.imwrite(os.path.join(image_name), frame)
        capture_start_time = None  # 캡처 완료 후 타이머 초기화
        print(f"이미지 성공적으로 저장. ({image_name})")
        yolo_result()
        return f'이미지 성공적으로 저장. ({image_name})'
    else:
        capture_start_time = None  # 캡처 실패 후 타이머 초기화
        print("이미지 저장 실패.")
        return '이미지 저장 실패.'

@app.route('/yolo_result')
def yolo_result():
    from roboflow import Roboflow
    rf = Roboflow(api_key="CwsFxkPSJLuJgcuN44Zw")
    project = rf.workspace().project("ingredients_detection")
    model = project.version(2).model
    print(model.predict("captured_image.jpg", confidence=40, overlap=30).json)
    model.predict("captured_image2.jpg", confidence=40, overlap=30).save("prediction.jpg")
if __name__ == '__main__':
    app.run(debug=True)
