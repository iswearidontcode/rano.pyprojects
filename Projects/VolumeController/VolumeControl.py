import cv2
import mediapipe as mp
import pyautogui
import math
import time

webcam = cv2.VideoCapture(0)
webcam.set(3, 1080)   # width
webcam.set(4, 540)   # height

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prev_volume = 0
last_action_time = 0
cooldown = 0.2  

while True:
    success, image = webcam.read()
    if not success:
        break

    frame_height, frame_width, _ = image.shape
    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )

            landmarks = hand.landmark

            x1 = int(landmarks[8].x * frame_width)  
            y1 = int(landmarks[8].y * frame_height)

            x2 = int(landmarks[4].x * frame_width) 
            y2 = int(landmarks[4].y * frame_height)

            cv2.circle(image, (x1, y1), 8, (0, 255, 255), -1)
            cv2.circle(image, (x2, y2), 8, (0, 0, 255), -1)

            distance = math.hypot(x2 - x1, y2 - y1)

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            current_time = time.time()

            if current_time - last_action_time > cooldown:
                if distance > 120:
                    pyautogui.press("volumeup")
                    last_action_time = current_time

                elif distance < 40:
                    pyautogui.press("volumedown")
                    last_action_time = current_time

    cv2.imshow("Volume Control", image)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

webcam.release()
cv2.destroyAllWindows()