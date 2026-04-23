import os; os.environ["MEDIAPIPE_DISABLE_GPU"]="1"
import cv2, mediapipe as mp
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 model_complexity=1, min_detection_confidence=0.6,
                                 min_tracking_confidence=0.7)
while True:
    ok, frame = cap.read()
    if not ok: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow("mp test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
