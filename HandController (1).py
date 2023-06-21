import json
import cv2
import mediapipe as mp
import redis
import numpy as np
from math import acos, sqrt, atan2, sin, cos, radians, degrees


SPEED_MAX = 0.3
D_THRES = 0.1
FREQ_SAMPLE = 20
ROLL_POINT_DEGREES = -90

RDS = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

def isStop(hand, thres):
    p8 = np.array([hand.landmark[8].x, hand.landmark[8].y, hand.landmark[8].z])
    p0 = np.array([hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z])
    if np.linalg.norm(p0-p8) < thres:
        return True
    return False

def Descartes2Polar(point):
    r = sqrt(point[0]**2 + point[1]**2)
    if r == 0:
        return 0, 0
    return r, atan2(point[1], point[0])
    # t = acos(point[0]/r)
    # if point[1] < 0:
    #     return r, -t
    # else:
    #     return r, t

def centerCrop(image, create_new = False):
    if create_new:
        image = image.copy()
    # Lấy kích thước ảnh ban đầu
    height, width = image.shape[0], image.shape[1]

    # Tính độ dài cạnh nhỏ nhất
    min_dim = min(height, width)

    # Tính tọa độ điểm giữa ảnh
    center_x, center_y = int(width/2), int(height/2)

    # Tính tọa độ và kích thước của vùng cắt
    crop_x = center_x - int(min_dim/2)
    crop_y = center_y - int(min_dim/2)

    # Thực hiện cắt ảnh
    image = image[crop_y:crop_y+min_dim, crop_x:crop_x+min_dim]

    return image

def rollPoint(point, center, angle):
    # Tính khoảng cách giữa point và center
    distance = sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
    
    # Tính góc của point và center
    theta = atan2(point[1]-center[1], point[0]-center[0])
    
    # Tính góc mới sau khi quay
    theta += radians(angle)
    
    # Tính tọa độ điểm mới sau khi quay
    x = center[0] + distance*cos(theta)
    y = center[1] + distance*sin(theta)
    
    return (x,y)

def sample(value, min, max, freq):
    n = (max - min)/freq
    return round(value/n)*n + min

def point_translation(point, vector):
    return point[0] + vector[0], point[1] + vector[1]

cap = cv2.VideoCapture(0)
size = int(min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    angle_pre = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = centerCrop(image)
        
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        v, angle = [0, 0]
        if results.multi_hand_landmarks:
            if not isStop(results.multi_hand_landmarks[0], 0.25):
                t = results.multi_hand_landmarks[0].landmark[8]
                # print(t)
                # point = (t.x, t.y, t.z)
                point = point_translation((t.x, t.y), (-0.5, -0.5)) 
                point = rollPoint(point, center=(0,0), angle=ROLL_POINT_DEGREES)
                r, a = Descartes2Polar(point)
                # v = 0
                if r > D_THRES:
                    v = SPEED_MAX
                    ang_s = sample(degrees(a), -180, 180, FREQ_SAMPLE) 
                    angle = -ang_s
                else:
                    angle = angle_pre
                    print("Stop")

                
                # RDS.set('controller', json.dumps([v, -ang_s]))
                
            else:
                # RDS.set('controller', json.dumps([0, 0]))
                angle = angle_pre
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        else: 
            # RDS.set('controller', json.dumps([0, 0]))
            angle = angle_pre
        RDS.set('controller', json.dumps([v, angle]))
        angle_pre = angle
        radius= int(D_THRES*size)
        cv2.circle(image,center= (size//2 , size//2), radius= radius, color=(0,0,255), thickness=2)
        cv2.imshow('Controller', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
zip()        
cap.release()