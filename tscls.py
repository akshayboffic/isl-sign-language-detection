import pickle
import cv2
import mediapipe as mp
import numpy as np

with open('isl_model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']


labels = model.classes_  
labels_dict = {i: label for i, label in enumerate(labels)}


cap = cv2.VideoCapture(0)  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        x1 = int(min(x_) * W) - 20
        y1 = int(min(y_) * H) - 20
        x2 = int(max(x_) * W) + 20
        y2 = int(max(y_) * H) + 20

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = prediction[0]
            predicted_character = predicted_label

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3, cv2.LINE_AA)
        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow('ISL Real-Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
