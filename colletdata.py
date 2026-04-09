import cv2
import mediapipe as mp # type: ignore
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect_gesture(gesture_name, num_samples=200, delay=2):
    save_path = f"data/{gesture_name}.csv"
    os.makedirs("data", exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" Kamera tidak terbuka.")
        return

    print(f" Kamera berhasil dibuka! Siap merekam gesture: '{gesture_name}'")
    print(f"Tunggu {delay} detik...")

    time.sleep(delay)
    print("🎬 Mulai rekam! Tekdan [Q] untuk berhenti lebih cepat.")

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    data = []

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Gagal membaca frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ambil koordinat landmark tangan
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append(landmarks)
                counter += 1
                print(f"\r📸 Sampel ke-{counter}/{num_samples}", end="")

                if counter >= num_samples:
                    print("\n Selesai merekam gesture!")
                    break

        cv2.imshow("Collecting Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n Dihentikan manual.")
            break

        if counter >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Simpannya di folder data csvnya
    if len(data) > 0:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f" Data disimpan ke: {save_path}")
    else:
        print("Tidak ada data yang disimpan.")

if __name__ == "__main__":
    gesture_name = input("Nama gesture: ").strip().lower()
    collect_gesture(gesture_name)
