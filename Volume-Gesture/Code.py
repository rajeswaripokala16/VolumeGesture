import cv2
import mediapipe as mp
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Try the standard GetSpeakers() and Activate:
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_min, vol_max = volume.GetVolumeRange()[:2]
except AttributeError:
    # Use the first available device that supports Activate
    found = False
    for device in AudioUtilities.GetAllDevices():
        try:
            interface = device.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            vol_min, vol_max = volume.GetVolumeRange()[:2]
            found = True
            print("Using fallback device:", device.FriendlyName)
            break
        except Exception:
            continue
    if not found:
        raise Exception("No suitable audio device found for volume control!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

def get_landmark_positions(hand_landmarks, w, h):
    return [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_landmarks.landmark)]

cap = cv2.VideoCapture(0)
print("Starting gesture-based volume control. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = get_landmark_positions(hand_landmarks, w, h)
            if len(lm_list) >= 9:
                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]
                cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                length = np.hypot(x2 - x1, y2 - y1)
                length = max(30, min(length, 200))
                vol = np.interp(length, [30, 200], [vol_min, vol_max])
                volume.SetMasterVolumeLevel(vol, None)
                vol_perc = int(np.interp(length, [30, 200], [0, 100]))
                cv2.putText(frame, f'Volume: {vol_perc}%', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting gesture volume control.")
        break

cap.release()
cv2.destroyAllWindows()
