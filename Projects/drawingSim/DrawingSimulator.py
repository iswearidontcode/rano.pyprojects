import cv2
import mediapipe as mp
import numpy as np

SHOW_LANDMARKS = True

def is_finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

def detect_gesture(landmarks):
    index = is_finger_up(landmarks, 8, 6)
    middle = is_finger_up(landmarks, 12, 10)
    ring = is_finger_up(landmarks, 16, 14)
    pinky = is_finger_up(landmarks, 20, 18)

    if index and not middle and not ring and not pinky:
        return "DRAW"
    if not index and not middle and not ring and not pinky:
        return "ERASE"
    return "NONE"

cv2.namedWindow("Controls")
cv2.createTrackbar("Thickness", "Controls", 5, 50, lambda x: None)
cv2.createTrackbar("Opacity", "Controls", 100, 100, lambda x: None)
cv2.createTrackbar("R", "Controls", 0, 255, lambda x: None)
cv2.createTrackbar("G", "Controls", 255, 255, lambda x: None)
cv2.createTrackbar("B", "Controls", 255, 255, lambda x: None)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)

canvas = None
prev = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    thickness = max(1, cv2.getTrackbarPos("Thickness", "Controls"))
    opacity = cv2.getTrackbarPos("Opacity", "Controls") / 100.0
    r = cv2.getTrackbarPos("R", "Controls")
    g = cv2.getTrackbarPos("G", "Controls")
    b = cv2.getTrackbarPos("B", "Controls")
    color = (b, g, r)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        if SHOW_LANDMARKS:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lms = hand.landmark
        gesture = detect_gesture(lms)

        h, w, _ = frame.shape
        x = int(lms[8].x * w)
        y = int(lms[8].y * h)

        if gesture == "DRAW":
            if prev is not None:
                cv2.line(canvas, prev, (x, y), color, thickness)
            prev = (x, y)

        elif gesture == "ERASE":
            cv2.circle(canvas, (x, y), thickness * 4, (0, 0, 0), -1)
            prev = None
        else:
            prev = None

        cv2.putText(frame, gesture, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    blended = cv2.addWeighted(frame, 1.0, canvas, opacity, 0)

    cv2.putText(
        blended,
        "H = Toggle Landmarks | C = Clear | X = Exit",
        (20, blended.shape[0]-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Drawing Simulator but its harder and buggier and idfk", blended)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('h'), ord('H')):
        SHOW_LANDMARKS = not SHOW_LANDMARKS
    elif key in (ord('c'), ord('C')):
        canvas[:] = 0
    elif key in (ord('x'), ord('X')):
        break

cap.release()
cv2.destroyAllWindows()
