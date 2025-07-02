import cv2
import mediapipe as mp
import time
import vlc

media_path = "/home/theertha/Desktop/hand_based/Eleven (2025) Tamil HQ HDRip - 1080p - HEVC - (DD+5.(2).mkv"
player = vlc.MediaPlayer(media_path)
player.play()
time.sleep(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def get_finger_states(hand_landmarks, handedness):
    fingers = []

    if handedness == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

 
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    return fingers

last_action = None
last_time = 0
gesture_count = 0
required_stable_frames = 5
prev_gesture = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = None

    if result.multi_hand_landmarks and result.multi_handedness:
        handLms = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0].classification[0].label
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        fingers = get_finger_states(handLms, handedness)

        if fingers == [1, 1, 1, 1, 1]:
            gesture = 'Play'
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = 'Pause'
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = '2x Speed'


        if gesture == prev_gesture:
            gesture_count += 1
        else:
            gesture_count = 1
        prev_gesture = gesture

        current_time = time.time()
        if gesture_count >= required_stable_frames and (gesture != last_action or current_time - last_time > 2):
            last_action = gesture
            last_time = current_time
            print(f"Gesture Triggered: {gesture}")

            if gesture == 'Play':
                player.set_rate(1.0)
                player.play()

            elif gesture == 'Pause':
                player.pause()

            elif gesture == '2x Speed':
                player.set_rate(2.0)
                
        if gesture:
            cv2.putText(frame, gesture, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
player.stop()
