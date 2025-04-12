import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Domain\Sign Language Detection\DATA-SCIENCE\HandSigns\SIGN LANG\allimgs'

data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    print(f"Processing: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(label)

with open('isl_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved to isl_data.pickle")