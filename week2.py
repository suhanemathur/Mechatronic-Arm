import cv2
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math
import threading

# Function to calculate angles between points
def calculate_angle(p1, p2, p3):
    a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    b = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    c = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    
    # Prevent potential division by zero or invalid cosine values
    try:
        # Ensure the value inside acos is within the valid range of -1 to 1
        cosine_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        cosine_angle = max(-1, min(1, cosine_angle))  # Clamp between -1 and 1
        angle = math.degrees(math.acos(cosine_angle))
        return angle
    except ZeroDivisionError:
        # Handle cases where the points are too close, or collinear
        return 0.0

# Function to check if the hand is in a fist
def is_fist(landmarks):
    finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb tip, Index tip, etc.
    palm_base = landmarks[0]  # Wrist landmark
    fist_threshold = 40  # Threshold for fist detection

    # Calculate the Euclidean distance between each finger tip and the palm base
    distances = [math.sqrt((tip[0] - palm_base[0]) ** 2 + (tip[1] - palm_base[1]) ** 2) for tip in finger_tips]
    
    # If all distances are below the threshold, classify it as a fist
    return all(dist < fist_threshold for dist in distances)

def lines_and_box(image, points):
    fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    angles = []
    
    if is_fist(points):
        cv2.putText(image, 'Fist Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for i, finger in enumerate(fingers):
            for j in range(len(finger) - 2):
                p1 = points[finger[j]]
                p2 = points[finger[j + 1]]
                p3 = points[finger[j + 2]]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
                cv2.putText(image, f'{angle:.1f}', (p2[0], p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw lines between points
            for j in range(len(finger) - 1):
                p1 = points[finger[j]]
                p2 = points[finger[j + 1]]
                cv2.line(image, p1, p2, (0, 255, 0), 2)
    
    return angles

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker")
        
        self.start_button = tk.Button(root, text="Start", command=self.start_tracking)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.cap = None
        self.csv_data = []
        self.running = False
        self.thread = None

    def start_tracking(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.thread = threading.Thread(target=self.capture_and_track)
        self.thread.start()

    def stop_tracking(self):
        self.running = False
        self.thread.join()  # Wait for the thread to finish
        self.cap.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close any OpenCV windows

        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.save_to_csv()

    def capture_and_track(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [
                        (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                        for lm in hand_landmarks.landmark
                    ]
                    
                    if landmarks and len(landmarks) >= 21:
                        angles = lines_and_box(frame, landmarks)
                        self.csv_data.append(angles)

            frame = cv2.flip(frame, 1)  # Optional: Flip the frame horizontally for a mirror view
            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Ensure the resources are released when the loop is done
        self.cap.release()
        cv2.destroyAllWindows()

    def save_to_csv(self):
        if self.csv_data:
            df = pd.DataFrame(self.csv_data, columns=[f'Finger_{i}_Joint_{j}_Angle' for i in range(1, 6) for j in range(1, 4)])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
            else:
                print("Save operation canceled.")
        else:
            print("No data to save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()
