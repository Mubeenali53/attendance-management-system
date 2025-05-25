import cv2 as cv
import numpy as np
import face_recognition
import cvzone
import pickle
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendance-fe76b-default-rtdb.firebaseio.com/"
})

cap = cv.VideoCapture(0)

file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()


# Path to the folder containing JPEG images
folder_path = "C:/Users/syedm/OneDrive/Desktop/Hackthon/Gestures/"

# Get a list of JPEG image file names in the folder
jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]


# Randomly select a JPEG image file
selected_image_file = random.choice(jpeg_files)

 # Construct the full path to the selected JPEG image
selected_image_path = os.path.join(folder_path, selected_image_file)

overlay_image = cv.imread(selected_image_path)
overlay_image = cv.imshow("photo", overlay_image)

# def map_distance_to_color(distance, max_distance):
#     # Map the distance to a value between 0 and 1
#     normalized_distance = min(distance / max_distance, 1.0)

#     # Map the normalized distance to a color (e.g., using a gradient)
#     color = (int(255 * (1 - normalized_distance)), int(255 * normalized_distance), 0)

#     return color

# def calculate_distance(bbox, frame_width, frame_height):
#     # Calculate the center of the bounding box
#     center_x = bbox[0] + bbox[2] // 2
#     center_y = bbox[1] + bbox[3] // 2

#     # Calculate the distance from the center to the frame center
#     distance = np.sqrt((center_x - frame_width // 2) ** 2 + (center_y - frame_height // 2) ** 2)

#     return distance

while True:
    ret, frame = cap.read()   
    faceCurFrame = face_recognition.face_locations(frame)
    encodeCurFrame = face_recognition.face_encodings(frame, faceCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        y1, x2, y2, x1 = faceLoc
        bbox = x1, y1, x2 - x1, y2 - y1
        id = studentIds[matchIndex]
        if matches[matchIndex]:
            name = studentIds[matchIndex]
            cv.putText(frame, name, (x1 + 70, y2 + 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            if name:
                cv.putText(frame,'Imitate the action', (150, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),1)
                x, y, c = frame.shape
                framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                result = hands.process(framergb)
                className = ''
                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)
                            landmarks.append([lmx, lmy])
                        #Drawing landmarks on frames
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                    if className == selected_image_path[50:-5]:
                        studentInfo = db.reference(f'Students Data/{id}').get()
                        datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],"%Y-%m-%d %H:%M:%S")
                        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                        print(secondsElapsed)
                        if secondsElapsed > 86400:
                            ref = db.reference(f'Students Data/{id}')
                            studentInfo['total_attendance'] += 1
                            ref.child('total_attendance').set(studentInfo['total_attendance'])
                            ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            print(studentInfo)
                            print('Attendance Marked')
                        else:
                           print(studentInfo) 
                cv.putText(frame, className, (10, 50), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv.LINE_AA)
            # Calculate distance between face and camera
            # distance = calculate_distance(bbox, frame.shape[1], frame.shape[0])
            # distance = abs(distance)
            # print(distance)

            # Map distance to color
            # color = map_distance_to_color(distance, max_distance=300)

            frame = cvzone.cornerRect(frame,bbox,rt=0)
        else:
            # color = map_distance_to_color(distance, max_distance=300)
            frame = cvzone.cornerRect(frame,bbox,rt=0)
            cv.putText(frame, 'unknown face', (x1 + 70, y2 + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)   
    cv.imshow('video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

