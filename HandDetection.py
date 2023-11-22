import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.85, min_tracking_confidence=0.75,
                      max_num_hands=2)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        if len(results.multi_handedness) == 2:
            cv2.putText(img, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 2)

            for h, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                label = h.classification[0].label

                palm_landmark = hand_landmarks.landmark[9]

                palm_x = int(palm_landmark.x * img.shape[1])
                palm_y = int(palm_landmark.y * img.shape[0])

                square_size = 50
                cv2.rectangle(img, (palm_x - square_size, palm_y - square_size),
                              (palm_x + square_size, palm_y + square_size),
                              (0, 255, 0), 2)

        else:
            for h in results.multi_handedness:
                label = h.classification[0].label

                hand_landmarks = results.multi_hand_landmarks[0]

                palm_landmark = hand_landmarks.landmark[9]

                palm_x = int(palm_landmark.x * img.shape[1])
                palm_y = int(palm_landmark.y * img.shape[0])

                square_size = 50
                cv2.rectangle(img, (palm_x - square_size, palm_y - square_size),
                              (palm_x + square_size, palm_y + square_size),
                              (0, 255, 0), 2)

                if label == 'Left':
                    cv2.putText(img, label + ' Hand', (250, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 0), 2)

                if label == 'Right':
                    cv2.putText(img, label + ' Hand', (250, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 0), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q') or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
