import cv2
import mediapipe as mp
import csv
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def initialize_csv(file_name):
    """Initialize a CSV file for saving lip movement data."""
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "landmark_x", "landmark_y"])

def save_landmarks(file_name, timestamp, landmarks):
    """Save lip landmarks to a CSV file."""
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        for landmark in landmarks:
            writer.writerow([timestamp, landmark[0], landmark[1]])

def detect_and_track_lips(word, output_file):
    """Detect and track lip movements."""
    cap = cv2.VideoCapture(0)
    print("Initializing camera... Position your face in the frame. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract lip landmarks
                landmarks = [
                    (int(face_landmarks.landmark[idx].x * frame.shape[1]),
                     int(face_landmarks.landmark[idx].y * frame.shape[0]))
                    for idx in [61, 62, 63, 64, 65, 66, 67, 68, 78, 80]
                ]
                
                # Save landmarks with timestamp
                save_landmarks(output_file, time.time(), landmarks)

                # Draw landmarks on frame
                for x, y in landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display instructions
        cv2.putText(frame, f"Say: '{word}'", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Lip Movement Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
