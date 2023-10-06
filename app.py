'''
Class Name or File Name: app.py
* Description: 클라이언트 서버 메인 파일로 웹페이지 백엔드 기능과 식재료 탐지 AI 호출, 레시피 추천 AI를 호출한다.
* Included Methods: 1. index()
                    2. capture()
                    3. yolo_result()
                    4. page_move()
                    5. recommend()

Author: Jeong Jae Min
Date : 2023-09-20
Version: release 1.0 on 2023-09-20
Change Histories: ("captureButton").addEventListener was updated by 정재민 2023-09-20.
       xhr.open("GET", "/capture", true); was updated by 노민성 2023-09-20.
       document.getElementById("pagemove").addEventListener by 이인규 2023-09-20.
'''

'''
1. Method Name: index()
* Function: 플라스크 웹 메인 페이지를 실행한다.
* Return Value: render_template('index.html') if it performs completely; an error code otherwise. '''

'''
2. Method Name: capture()
* Function: 웹 캐에서 냉장고 안에 이미지를 캡처하고 식재료 탐지 AI를 호출한뒤 탐지된 식재료 목록들을 반환한다.
* Return Value: return jsonify(classes) if it performs completely; an error code otherwise. '''

'''
3. Method Name: yolo_result()
* Function: 캡처된 냉장고 이미지를 식재료 탐지 AI를 이용하여 식재료들을 탐지한다. 
            탐지된 식재료를 호출한 함수에 반환한다.
* Return Value: new_classes if it performs completely; an error code otherwise. '''

'''
4. Method Name: page_move()
* Function: 레시피 추천 웹페이지로 페이지를 이동한다.
* Return Value: render_template('recommend.html', ingredients=new_classes) if it performs completely; an error code otherwise. '''

'''
5. Method Name: recommend()
* Function: 식재료 탐지 AI를 호출하고 추천받은 레시피를 recommend.html에 전송한다.
* Parameter: table=추천 받은 레시피를 테이블 형태.
             ingredients= 레시피에 사용된 식재료의 정보.
* Return Value: render_template('recommend.html',table=html_table,ingredients=new_classes) if it performs completely; an error code otherwise. '''



from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import sys, os
from recommend import based_Ingredient
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
capture_start_time = None
camera = None  # 웹캠 객체를 전역 변수로 선언

ingredient_classes={"egg":"달걀", "garlic":"마늘","greenonion":"파","lettuce":"상추","meat":"고기","onion":"양파","tofu":"두부"}
global html_table
global new_classes
# 웹캠으로부터 프레임을 가져오는 함수
def get_frame():
    global camera  # 전역 변수 사용

    camera = cv2.VideoCapture(0) #첫번째 카메라를 객체로 가져옴
    os.remove("./static/prediction.jpg")
    
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

def dataframe_to_html(dataframe):
    return dataframe.to_html(classes='table table-striped', index=False)

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
        print(f"이미지 성공적으로 저장. ({image_name})")

        classes=yolo_result()

        print(classes)

        return jsonify(classes)
    else:
        print("이미지 저장 실패.")

def yolo_result():
    global new_classes
    from roboflow import Roboflow
    rf = Roboflow(api_key="CwsFxkPSJLuJgcuN44Zw")
    project = rf.workspace().project("ingredients_detection")
    model = project.version(2).model
    model.predict("captured_image.jpg", confidence=40, overlap=30).save("./static/prediction.jpg")
    #pred 에 x y width height confidence class image_path prediction_type 이 있다
    pred = model.predict("captured_image.jpg", confidence=40, overlap=30)
    new_classes = []
    #class 목록까지 불러옴
    classes = [prediction['class'] for prediction in pred]
    for c in classes:
        new_classes.append(ingredient_classes[c])
    return new_classes


@app.route('/page_move',methods=('POST','GET'))
def page_move():
    print("classes", new_classes)
    return render_template('recommend.html', ingredients=new_classes)

@app.route('/recommend_page',methods=['GET'])
def recommend_page():
    return render_template('recommend.html',table=html_table,ingredients=new_classes)

#선택된 식재료를 이용하여 레시피 추천하는 함수 호출하기
@app.route('/recommend',methods=('POST','GET'))
def recommend():
    selected_ingredients = request.json.get('dataArray')
    print("selected_ingredients", selected_ingredients)
    input = ', '.join(selected_ingredients)
    rec = based_Ingredient.get_recs(input)
    # 이 예시에서는 간단히 선택된 재료를 출력합니다.
    print(rec)
    global html_table
    html_table = dataframe_to_html(rec)
    return render_template('recommend.html',table=html_table,ingredients=new_classes)


if __name__ == '__main__':
    app.run(debug=True)