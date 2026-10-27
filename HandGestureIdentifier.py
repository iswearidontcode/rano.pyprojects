import cv2
import mediapipe as mp
import time
SHOW_LANDMARKS = True
def is_finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y
def is_thumb_up(landmarks, handedness="Right"):
    # Thumb detection varies by hand
    if handedness == "Right":
        return landmarks[4].x > landmarks[3].x
    else:
        return landmarks[4].x < landmarks[3].x
def get_finger_states(landmarks, handedness="Right"):
    return {
        "thumb": is_thumb_up(landmarks, handedness),
        "index": is_finger_up(landmarks, 8, 6),
        "middle": is_finger_up(landmarks, 12, 10),
        "ring": is_finger_up(landmarks, 16, 14),
        "pinky": is_finger_up(landmarks, 20, 18),
    }
def fingers_touching(landmarks, tip1, tip2, threshold=0.05):
    """Check if two fingertips are close together."""
    dx = landmarks[tip1].x - landmarks[tip2].x
    dy = landmarks[tip1].y - landmarks[tip2].y
    distance = (dx**2 + dy**2) ** 0.5
    return distance < threshold
def detect_gesture(landmarks, handedness="Right"):
    f = get_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = (
        f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]
    )
    # Count extended fingers
    extended_count = sum([index, middle, ring, pinky])
    
    if fingers_touching(landmarks, 4, 8, 0.06) and middle and ring and pinky:
        return "THREE", (0, 255, 200)

    if index and middle and not ring and not pinky:
        spread = abs(landmarks[8].x - landmarks[12].x) > 0.05
        if spread:
            return "PEACE", (255, 200, 0)
        else:
            return "TWO", (200, 200, 0)

    if index and pinky and not middle and not ring:
        return "ROCK ON", (255, 0, 100)

    if thumb and pinky and not index and not middle and not ring:
        return "CALL ME", (100, 255, 100)

    if thumb and index and pinky and not middle and not ring:
        return "I LOVE YOU", (255, 100, 255)

    if thumb and not index and not middle and not ring and not pinky:
        return "THUMBS UP", (0, 255, 0)
    
    if (not index and not middle and not ring and not pinky and
            landmarks[4].y > landmarks[3].y):
        return "THUMBS DOWN", (0, 100, 255)

    if index and not middle and not ring and not pinky:
        return "POINTING / ONE", (255, 255, 0)

    if middle and not index and not ring and not pinky:
        return "MIDDLE FINGER", (0, 0, 255)

    if index and middle and ring and not pinky:
        return "THREE", (255, 150, 0)

    if index and middle and ring and pinky and not thumb:
        return "FOUR", (150, 255, 0)

    if index and middle and ring and pinky:
        return "OPEN HAND / FIVE", (255, 0, 255)

    if not index and not middle and not ring and not pinky:
        return "FIST / ZERO", (0, 165, 255)
    
    if pinky and not index and not middle and not ring:
        return "PINKY", (255, 180, 200)
    return "PINKY PROMISE", (200, 200, 200)

webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
prev_time = 0

while True:
    success, image = webcam.read()
    if not success:
        break
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand, hand_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            
            handedness = hand_info.classification[0].label
            handedness = "Left" if handedness == "Right" else "Right"
            if SHOW_LANDMARKS:
                mp_draw.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            landmarks = hand.landmark
            x_list = [int(lm.x * frame_width) for lm in landmarks]
            y_list = [int(lm.y * frame_height) for lm in landmarks]
            x_min = max(min(x_list) - 30, 0)
            y_min = max(min(y_list) - 30, 0)
            x_max = min(max(x_list) + 30, frame_width)
            y_max = min(max(y_list) + 30, frame_height)
            gesture, color = detect_gesture(landmarks, handedness)
            # Draw semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            image = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)
            # Draw border
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
            # Draw label background and text
            label = f"{handedness}: {gesture}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                image,
                (x_min, y_min - 35),
                (x_min + text_size[0] + 20, y_min),
                color, -1
            )
            cv2.putText(
                image, label, (x_min + 10, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

    current_time = time.time()
    fps = int(1 / (current_time - prev_time + 0.0001))
    prev_time = current_time
    cv2.putText(image, f"FPS: {fps}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, "H = Toggle Landmarks | X = Exit",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Hand Gesture Recognition", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('h') or key == ord('H'):
        SHOW_LANDMARKS = not SHOW_LANDMARKS
    if key == ord('x'):
        break
webcam.release()
cv2.destroyAllWindows()