# AI_Health_Trainer

졸업작품으로 진행했던 AI Health Trainer 모델입니다.

머신러닝을 활용하여 운동 자세 추정 및 자세 교정 제안 모델을 구축하였습니다.

---
### 데이터 수집

사용한 데이터는 스쿼트 동작을 취하고 있는 사람의 옆모습을 촬영한 이미지 데이터 입니다.

* 좋은 스쿼트 자세: 1024장

* 나쁜 스쿼트 자세: 1034장

총 2058장의 이미지 데이터를 사용하였습니다.

---
### 관절 추출

각 관절의 좌표값을 추출하는 과정입니다.

Mediapipe Pose를 활용하여 33개의 랜드마크로 이미지 속 자세를 추정하였습니다.

```
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
```

각 관절의 좌표값을 통해 6가지의 각도 변수를 생성하였습니다.

* 목 각: 귀 - 어깨 - 지면

* 엉덩이 각: 어깨 - 엉덩이 - 무릎

* 허리 각: 어깨 - 엉덩이 - 지면

* 무릎 각: 엉덩이 - 무릎 - 발목

* 발목 각: 무릎 - 발목 - 발 앞꿈치

* 발 뒤꿈치 각: 발목 - 발 뒤꿈치 - 지면

다음은 좌표들을 통해 각도를 계산하는 함수입니다.

```
# define "calculateAngle" model
    # Args :
        ## landmark1 : 1st (x,y,z)
        ## landmark2 : 2nd (x,y,z)
        ## landmark3 : 3rd (x,y,z)
    # Returns :
        ## angle : between 1,2,3
        
def calculateAngle(landmark1, landmark2, landmark3):

    # get landmarks
    x1, y1= landmark1
    x2, y2= landmark2
    x3, y3= landmark3

    # calculatate with math
    # math.atan2(y, x) : 탄젠트의 역함수. atan과 다르게 atan2는 x축으로부터 반시계방향으로 각도를 계산. 이후 라디안 값으로 뱉어냄
    # math.degrees : 라디안값을 각도로 변경해주는 함수
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # find opposite angle
    if angle < 0.0:
        angle += 360
    if angle >180.0:
        angle = 360.0 - angle
  
    # iterate
    return angle
```
```
# 두 측정점과 지면과의 각도 구하는 함수

def calculateAngle2(landmark1, landmark2):

    # get landmarks
    x1, y1= landmark1
    x2, y2= landmark2
    x3 = x2
    y3 = y2 + 1
    
    # calculate with math
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # find opposite angle
    if angle < 0.0:
        angle += 360
    if angle >180.0:
        angle = 360.0 - angle
  
    # iterate
    return angle
```
---
### 모델 학습

분류모델 학습을 위해 데이터셋에 라벨링을 진행하였습니다.

* 나쁜 스쿼트 자세 Y = 0

* 좋은 스쿼트 자세 Y = 1

모델은 Logistic Regression, Decision Tree, Random Forest, Gradient Boosting 4가지를 학습시키고 결과 값을 비교하였습니다.

4가지 모델 중 Random Forest를 사용하여 분류 및 실시간 영상에서의 자세 분류를 진행하기로 결정했습니다.

```
RF = RandomForestClassifier(random_state = 0, criterion = 'gini', max_depth = None, n_estimators= 59, max_features='sqrt')
RF.fit(X_train, Y_train)
```
---
### 자세 교정 제안

앞서 구축한 모델을 이용하여 영상에서의 스쿼트 자세를 분석 후 자세의 좋고 나쁨을 구별하고, 스쿼트 자세가 나쁠 경우 어느 부분을 교정해야 하는지 화면상에서 피드백을 줄 수 있도록 구현하였습니다.

```
if len(angle) != 0:
        
        r_knee_angle = angle[0]
        r_hip_angle = angle[1] 
        r_ankle_angle = angle[2]
        
        if r_knee_angle < 140:
            if check_point == 0:
                if (r_knee_angle > previous_knee_angle) & (r_hip_angle > previous_hip_angle) & (r_ankle_angle > previous_ankle_angle):
                    state = 'squat'
                    pred = load_model.predict([x_input])
                    result.append(x_input)
                    dd += 1
                    if pred == 0:
                        cla = 'CLASS : BAD'
                        if x_input[4] < 132: #132
                            comment.append('Raise your neck more')
                            cc += 1
                        if x_input[0] > 65: #79
                            comment.append('Sit down more')
                            cc += 1
                        if x_input[3] < 125: #125
                            comment.append('Lift your upper body')
                            cc += 1
                        if x_input[5] > 100: #100
                            comment.append('Pull your knee back')
                            cc+= 1
                    else:
                        cla = 'CLASS : GOOD' 
            
                    check_point = 1
                            
    previous_knee_angle = r_knee_angle
    previous_hip_angle = r_hip_angle
    previous_ankle_angle = r_ankle_angle
    x_input = angle
    
    if (state == 'squat') & (r_knee_angle > 155):
        if cla == 'CLASS : GOOD':
            count += 1
        check_point = 0
        state = 'stand'
        cla = 'CLASS : None'
        comment = []
        cc = 0
        
        if dd == user_input:
            video.release()
            cv2.destroyAllWindows()
```
---
### 자세 패턴 분석

마할라노비스 거리를 활용하여 좋은 자세로 일관성 있게 스쿼트를 진행했는지 패턴을 분석하였습니다.

또한 각 변수를 제외하고 마할라노비스 값을 측정하여 자세의 이상 원인을 분석하였습니다.
