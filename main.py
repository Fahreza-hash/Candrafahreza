# bukan untuk development, hanya untuk praktikum atau uji coba atau tuagas kuliah dll

import cv2
import mediapipe as mp # type: ignore
import numpy as np
import joblib # type: ignore
import time
from src.speaker import Speaker

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model hasil training lu
model = joblib.load("models/gesture_model.pkl")
print(model.classes_)

# panggil data yang dilatih tadi
gesture_to_text = {
    "hallo":"hallo",
    "nama":"eko bilal saputra",
    "gerakan_hallo":"ini gerakan hallooooo",
    "gerakan_anjay":"anjayyyyy ayo kita mulai "
}

def predict_gesture(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)
    prediction = model.predict(landmarks)[0]
    return prediction

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    speaker = Speaker()
    prev_gesture = None
    gesture_timer = 0
    gesture_streak = 0
    last_gesture = None

    print("🎥 Kamera aktif! Tekan [Q] untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Gagal membaca kamera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                gesture = predict_gesture(landmarks)

                cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Biar gak spam suara (tracking gesture stabil)
                if gesture == last_gesture:
                    gesture_streak += 1
                else:
                    gesture_streak = 0
                    last_gesture = gesture

                # Cuma bicara kalau gesture stabil minimal 10 frame (~0.3 detik)
                if gesture_streak > 10 and (gesture != prev_gesture or (time.time() - gesture_timer) > 2):
                    if gesture in gesture_to_text:
                        speaker.say(gesture_to_text[gesture])
                    prev_gesture = gesture
                    gesture_timer = time.time()
                    gesture_streak = 0  

        cv2.imshow("AI Gesture to Voice", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    speaker.stop()
    print(" Program selesai.")

if __name__ == "__main__":
    main()
