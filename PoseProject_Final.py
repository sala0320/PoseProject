
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


def calculate_angle(a,b):
    a = np.array(a)
    b = np.array(b)
    
    radians = np.arctan2(b[1]-a[1], b[0]-a[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle 

def run():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    warning = False
    count = 0
    good_count = 0
    stretch_count = 0
    stand_count = 0
    start = time.gmtime(time.time())     # 시작 시간 저장

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            resize_frame = cv2.resize(frame ,None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) 
            
            # Recolor Feed
            image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
            # Export coordinates
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Calculate angle
                angle = calculate_angle(left_shoulder, right_shoulder)

                # Curl counter logic
                if angle < 175 or body_language_class.split(' ')[0] == 'Bad':
                    count = count + 1
                    good_count = 0   
                    
                elif angle >= 175:
                    good_count = good_count + 1

                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # Get status box
                cv2.rectangle(image, (0,0), (1000, 80), (128,128,128), -1)
                
                #Time
                now = time.gmtime(time.time())
            
                cv2.putText(image, 'Time', 
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                hour = now.tm_hour - start.tm_hour
                minutes = abs(now.tm_min - start.tm_min)
                cv2.putText(image, str(hour) +' : '+ str(minutes), 
                            (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                if minutes >= 5:
                    if body_language_class.split(' ')[0] == 'Stand' and round(body_language_prob[np.argmax(body_language_prob)],2) > 0.5:
                        stand_count += 1
                    if stand_count < 30:
                        cv2.putText(image, 'Please Stand UP', 
                            (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, 'Great!!', 
                                (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        stand_count = 0
                        count = 0
                        stretch_count = 0
                        good_count = 0
                        start = time.gmtime(time.time())  
                        time.sleep(0.1)
                 
                #Warning
                if minutes < 5 and count > 10:
                    if body_language_class.split(' ')[0] == 'Stretch' and round(body_language_prob[np.argmax(body_language_prob)],2) > 0.2:
                        stretch_count += 1
                        
                    if good_count > 5 or stretch_count > 20:
                        cv2.putText(image, 'Great!!', 
                                (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        count = 0
                        stretch_count = 0
                        good_count = 0
                        time.sleep(0.5)
                        
                    else:
                        cv2.putText(image, 'Please Straighten UP', 
                                    (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)  
                        
                # Display Class
                cv2.putText(image, 'Status'
                            , (150,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (150,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (280,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, str(round(angle,2))
                            , (850,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass
                            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    run()


